#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    :
# @File    : model_training.py
# @Version：V 0.1
# @desc :
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin


# 自定义加权集成模型类（符合scikit-learn接口）
class WeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models  # 基础模型字典
        self.weights = weights  # 权重字典
        self.trained_models = {}  # 存储训练后的模型

    def fit(self, X, y):
        # 训练所有基础模型
        for name, model in self.models.items():
            self.trained_models[name] = model.fit(X, y)
        return self

    def predict(self, X):
        # 加权集成预测
        predictions = np.zeros(len(X))
        for name, model in self.trained_models.items():
            predictions += self.weights[name] * model.predict(X)
        return predictions


# --------------------------
# 1. 加载数据
# --------------------------
def load_processed_data():
    """加载处理后的训练集和测试集"""
    try:
        train_df = pd.read_csv('processed_data/train_processed.csv')
        test_df = pd.read_csv('processed_data/test_processed.csv')
        print(f"训练集形状: {train_df.shape}")
        print(f"测试集形状: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError:
        print("错误：未找到处理后的数据文件，请先运行特征工程代码")
        exit()


# --------------------------
# 2. 数据准备
# --------------------------
train_processed, test_processed = load_processed_data()

# 分离特征和目标变量
X_train = train_processed.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
y_train = np.log1p(train_processed['SalePrice'])  # 对目标变量进行对数转换

# 测试集特征（不含目标变量）
# 处理测试集可能没有SalePrice列的情况
if 'SalePrice' in test_processed.columns:
    X_test = test_processed.drop(['Id', 'SalePrice'], axis=1)
else:
    X_test = test_processed.drop('Id', axis=1)
test_ids = test_processed['Id']  # 保存ID用于最终提交

# --------------------------
# 3. 模型定义与参数调优
# --------------------------
# 定义交叉验证策略（5折交叉验证，确保随机性）
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# 评估指标：RMSE（针对对数转换后的目标变量）
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=kf
    ))
    return rmse.mean(), rmse.std()


# 3.1 定义基础模型（带初步调优参数）
models = {
    # 线性模型（使用RobustScaler处理异常值）
    'Lasso': make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=10000, random_state=42)),
    'Ridge': make_pipeline(RobustScaler(), Ridge(alpha=5.0, random_state=42)),
    'ElasticNet': make_pipeline(RobustScaler(),
                                ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=42)),

    # 树模型
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
}

# 3.2 基础模型评估
print("基础模型交叉验证结果：")
model_scores = {}
for name, model in models.items():
    mean_rmse, std_rmse = rmse_cv(model, X_train, y_train)
    model_scores[name] = mean_rmse
    print(f"{name}: {mean_rmse:.4f} (±{std_rmse:.4f})")

# 3.3 自动参数调优（以表现较好的XGBoost为例）
print("\n开始XGBoost参数调优...")
xgb_param_grid = {
    'n_estimators': [800, 1000, 1200],
    'learning_rate': [0.02, 0.03],
    'max_depth': [3, 4, 5]
}

xgb_grid = GridSearchCV(
    XGBRegressor(subsample=0.9, colsample_bytree=0.9, random_state=42),
    param_grid=xgb_param_grid,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

print(f"最佳XGBoost参数: {xgb_grid.best_params_}")
print(f"最佳XGBoost交叉验证RMSE: {np.sqrt(-xgb_grid.best_score_):.4f}")

# 更新最佳单模型
best_single_model = xgb_grid.best_estimator_

# --------------------------
# 4. 模型集成
# --------------------------
# 4.1 简单加权集成（基于交叉验证分数分配权重）
print("\n构建加权集成模型...")
# 计算权重（分数越低，权重越高）
total_score = sum(1 / score for score in model_scores.values())
weights = {name: (1 / score) / total_score for name, score in model_scores.items()}

# 创建自定义加权集成模型
weighted_ensemble = WeightedEnsemble(models=models, weights=weights)

# 评估集成模型
ensemble_rmse, ensemble_std = rmse_cv(weighted_ensemble, X_train, y_train)
print(f"加权集成模型交叉验证RMSE: {ensemble_rmse:.4f} (±{ensemble_std:.4f})")

# 4.2 堆叠集成（Stacking）
print("\n构建堆叠集成模型...")
# 定义元模型（最终整合器）
meta_model = Ridge(alpha=10.0, random_state=42)

# 构建堆叠模型
stacking_model = StackingRegressor(
    estimators=[(name, model) for name, model in models.items()],
    final_estimator=meta_model,
    cv=kf,
    n_jobs=-1
)

# 评估堆叠模型
stacking_rmse, stacking_std = rmse_cv(stacking_model, X_train, y_train)
print(f"堆叠集成模型交叉验证RMSE: {stacking_rmse:.4f} (±{stacking_std:.4f})")

# 选择表现最好的模型作为最终模型
if stacking_rmse < ensemble_rmse and stacking_rmse < np.sqrt(-xgb_grid.best_score_):
    final_model = stacking_model
    final_model_type = "堆叠集成"
elif ensemble_rmse < np.sqrt(-xgb_grid.best_score_):
    final_model = weighted_ensemble
    final_model_type = "加权集成"
else:
    final_model = best_single_model
    final_model_type = "XGBoost"

# 训练最终模型（全量数据）
print(f"\n使用全量训练数据训练最终模型: {final_model_type}...")
final_model.fit(X_train, y_train)

# --------------------------
# 5. 预测与结果保存
# --------------------------
# 创建输出目录
os.makedirs('predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 保存模型
joblib.dump(final_model, 'models/final_model0.pkl')
print("最终模型已保存至 models/final_model0.pkl")

# 生成测试集预测（转换回原始价格尺度）
print("生成测试集预测...")
test_predictions = np.expm1(final_model.predict(X_test))

# 生成提交文件
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

# 确保ID排序正确
submission = submission.sort_values('Id').reset_index(drop=True)

# 保存为CSV文件，不包含索引
submission.to_csv('predictions/submission.csv', index=False)
print("预测结果已保存至 predictions/submission.csv")

# 输出最终模型性能对比
print("\n模型性能总结：")
print(f"最佳单模型 (XGBoost) RMSE: {np.sqrt(-xgb_grid.best_score_):.4f}")
print(f"加权集成模型 RMSE: {ensemble_rmse:.4f}")
print(f"堆叠集成模型 RMSE: {stacking_rmse:.4f}")
print(f"最终选择模型: {final_model_type}")

# 显示提交文件的前几行
print("\n提交文件前10行预览：")
print(submission.head(10))
"""
训练集形状: (1460, 256)
测试集形状: (1459, 256)
基础模型交叉验证结果：
Lasso: 0.0453 (±0.0080)
Ridge: 0.0478 (±0.0079)
ElasticNet: 0.0453 (±0.0080)
RandomForest: 0.0652 (±0.0123)
GradientBoosting: 0.0430 (±0.0060)
XGBoost: 0.0440 (±0.0076)
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001206 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.030658
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001310 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 168
[LightGBM] [Info] Start training from score 12.016898
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001627 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3259
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.022759
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001431 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3241
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 166
[LightGBM] [Info] Start training from score 12.027933
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001670 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3252
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 163
[LightGBM] [Info] Start training from score 12.022040
LightGBM: 0.0546 (±0.0060)

开始XGBoost参数调优...
Fitting 5 folds for each of 18 candidates, totalling 90 fits
最佳XGBoost参数: {'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 1200}
最佳XGBoost交叉验证RMSE: 0.0406

构建加权集成模型...
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000788 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.030658
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001514 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 168
[LightGBM] [Info] Start training from score 12.016898
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001389 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3259
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.022759
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001277 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3241
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 166
[LightGBM] [Info] Start training from score 12.027933
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001011 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3252
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 163
[LightGBM] [Info] Start training from score 12.022040
加权集成模型交叉验证RMSE: 0.0372 (±0.0075)

构建堆叠集成模型...
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001473 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.030658
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001245 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2981
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 154
[LightGBM] [Info] Start training from score 12.031717
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002848 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2997
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 153
[LightGBM] [Info] Start training from score 12.030567
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003700 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2985
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 155
[LightGBM] [Info] Start training from score 12.030043
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003631 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2970
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 155
[LightGBM] [Info] Start training from score 12.032550
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.004564 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2980
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 155
[LightGBM] [Info] Start training from score 12.028414
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001128 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 168
[LightGBM] [Info] Start training from score 12.016898
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002643 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2968
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 155
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003080 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Start training from score 12.024419
[LightGBM] [Info] Total Bins 2979
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 155
[LightGBM] [Info] Start training from score 12.021841
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003965 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2977
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 155
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.003095 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.[LightGBM] [Info] 
Start training from score 12.017805
[LightGBM] [Info] Total Bins 2964
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 153
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003968 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Start training from score 12.005874
[LightGBM] [Info] Total Bins 2947
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 154
[LightGBM] [Info] Start training from score 12.014565
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001475 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3259
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.022759
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001249 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2960
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 154
[LightGBM] [Info] Start training from score 12.021543
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.004684 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2970
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 154
[LightGBM] [Info] Start training from score 12.009874
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003974 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2963
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.005241 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 154
[LightGBM] [Info] Total Bins 2974
[LightGBM] [Info] Start training from score 12.024738
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 154
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.004925 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Start training from score 12.023955
[LightGBM] [Info] Total Bins 2975
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 154
[LightGBM] [Info] Start training from score 12.033683
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001191 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3241
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 166
[LightGBM] [Info] Start training from score 12.027933
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001172 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2973
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 156
[LightGBM] [Info] Start training from score 12.013311
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003876 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2967
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 153
[LightGBM] [Info] Start training from score 12.021757
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003039 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2956
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 153
[LightGBM] [Info] Start training from score 12.039155
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.003553 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2952
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 156
[LightGBM] [Info] Start training from score 12.034137
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.003798 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2973
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 154
[LightGBM] [Info] Start training from score 12.031284
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001558 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3252
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 163
[LightGBM] [Info] Start training from score 12.022040
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.012659 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.011075 seconds.
You can set `force_col_wise=true` to remove the overhead.[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.014779 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.


[LightGBM] [Info] Total Bins 2967
[LightGBM] [Info] [LightGBM] [Info] Total Bins 2987
Total Bins 2963
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 154
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 153
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 153
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.014243 seconds.
You can set `force_col_wise=true` to remove the overhead.[LightGBM] [Info] 
Start training from score 12.022144
[LightGBM] [Info] Start training from score 12.026399
[LightGBM] [Info] [LightGBM] [Info] Start training from score 12.028139
Total Bins 2982
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.012375 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Number of data points in the train set: 935, number of used features: 154
[LightGBM] [Info] Total Bins 2955
[LightGBM] [Info] Number of data points in the train set: 934, number of used features: 153
[LightGBM] [Info] Start training from score 12.017557
[LightGBM] [Info] Start training from score 12.015958
堆叠集成模型交叉验证RMSE: 0.0363 (±0.0076)

使用全量训练数据训练最终模型: 堆叠集成...
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.004475 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3477
[LightGBM] [Info] Number of data points in the train set: 1460, number of used features: 171
[LightGBM] [Info] Start training from score 12.024057
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines

[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.091529 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.030658
[LightGBM] [Info] [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.150976 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3252
Auto-choosing col-wise multi-threading, the overhead of testing was 0.150984 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 3260
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 168
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 163
[LightGBM] [Info] Start training from score 12.016898
[LightGBM] [Info] Start training from score 12.022040
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.069798 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3241
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 166
[LightGBM] [Info] Start training from score 12.027933
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.108156 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3259
[LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 164
[LightGBM] [Info] Start training from score 12.022759
最终模型已保存至 models/final_model0.pkl
生成测试集预测...
预测结果已保存至 predictions/submission.csv

模型性能总结：
最佳单模型 (XGBoost) RMSE: 0.0406
加权集成模型 RMSE: 0.0372
堆叠集成模型 RMSE: 0.0363
最终选择模型: 堆叠集成
"""