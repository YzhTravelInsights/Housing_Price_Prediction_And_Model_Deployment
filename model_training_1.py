#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    :
# @File    : model_training_1.py
# @Version：V 0.1
# @desc :
import pandas as pd
import numpy as np
import os
import joblib
import time
import logging
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.impute import SimpleImputer
import warnings
from visualization import HousingDataVisualizer

warnings.filterwarnings('ignore')

# --------------------------
# 设置日志和随机种子
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SEED = 42
np.random.seed(SEED)


# --------------------------
# 自定义特征选择器（处理特征对齐问题）
# --------------------------
class FeatureSelector(TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.selected_features = None

    def fit(self, X, y=None):
        self.selected_features = [f for f in self.features if f in X.columns]
        return self

    def transform(self, X):
        # 添加缺失的特征并填充为0
        for f in self.selected_features:
            if f not in X.columns:
                X[f] = 0

        # 移除多余的特征
        return X[self.selected_features]


# --------------------------
# 自定义加权集成模型类
# --------------------------
class WeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        self.trained_models = {}

    def fit(self, X, y):
        for name, model in self.models.items():
            logger.info(f"训练 {name}...")
            start_time = time.time()
            self.trained_models[name] = model.fit(X, y)
            logger.info(f"{name} 训练完成，耗时: {time.time() - start_time:.2f}秒")
        return self

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        total_weight = sum(self.weights.values())

        for name, model in self.trained_models.items():
            pred = model.predict(X)
            predictions += self.weights[name] * pred / total_weight

        return predictions


# --------------------------
# 1. 数据加载与验证
# --------------------------
def load_processed_data():
    """加载处理后的训练集和测试集"""
    try:
        train_df = pd.read_csv('processed_data/train_processed.csv')
        test_df = pd.read_csv('processed_data/test_processed.csv')
        logger.info(f"训练集形状: {train_df.shape}")
        logger.info(f"测试集形状: {test_df.shape}")

        # 验证数据完整性
        if 'SalePrice' not in train_df.columns:
            raise ValueError("训练集中缺少目标变量 'SalePrice'")

        return train_df, test_df
    except FileNotFoundError:
        logger.error("未找到处理后的数据文件，请先运行特征工程代码")
        raise


# --------------------------
# 2. 数据准备
# --------------------------
try:
    train_processed, test_processed = load_processed_data()

    # 分离特征和目标变量
    X_train = train_processed.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
    y_train = np.log1p(train_processed['SalePrice'])  # 对目标变量进行对数转换

    # 处理测试集
    if 'SalePrice' in test_processed.columns:
        X_test = test_processed.drop(['Id', 'SalePrice'], axis=1)
    else:
        X_test = test_processed.drop('Id', axis=1)

    test_ids = test_processed['Id']

    # 保存特征列表用于后续部署
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')

except Exception as e:
    logger.error(f"数据准备阶段出错: {e}")
    exit()


# --------------------------
# 3. 模型定义与评估
# --------------------------
# 定义评估指标
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

# 定义交叉验证策略
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 定义基础模型（使用Pipeline确保数据预处理一致性）
models = {
    'Lasso': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('model', Lasso(alpha=0.0005, max_iter=10000, random_state=SEED))
    ]),
    'Ridge': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('model', Ridge(alpha=5.0, random_state=SEED))
    ]),
    'RandomForest': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('model', RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=10,
            random_state=SEED, n_jobs=-1
        ))
    ]),
    'GradientBoosting': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('model', GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=SEED
        ))
    ]),
    'XGBoost': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('model', XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED
        ))
    ]),
    'LightGBM': Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('model', LGBMRegressor(
            n_estimators=1000, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED
        ))
    ])
}

# 评估基础模型
logger.info("开始评估基础模型...")
model_scores = {}
model_stds = {}

for name, model in models.items():
    logger.info(f"评估 {name}...")
    start_time = time.time()

    try:
        scores = cross_val_score(model, X_train, y_train,
                                 scoring=rmse_scorer, cv=kf, n_jobs=-1)
        mean_rmse = -scores.mean()
        std_rmse = scores.std()

        model_scores[name] = mean_rmse
        model_stds[name] = std_rmse

        logger.info(f"{name}: RMSE = {mean_rmse:.4f} (±{std_rmse:.4f}), 耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"评估 {name} 时出错: {e}")
        model_scores[name] = float('inf')
        model_stds[name] = 0

# --------------------------
# 4. 超参数调优
# --------------------------
logger.info("\n开始超参数调优...")

# 选择表现最好的2-3个模型进行调优
best_models = sorted(model_scores.items(), key=lambda x: x[1])[:3]
logger.info(f"选择进行调优的模型: {[name for name, _ in best_models]}")

# XGBoost参数调优
if 'XGBoost' in dict(best_models):
    logger.info("调优 XGBoost...")

    xgb_param_dist = {
        'model__n_estimators': [800, 1000, 1200],
        'model__learning_rate': [0.01, 0.02, 0.03],
        'model__max_depth': [3, 4, 5],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0]
    }

    xgb_model = models['XGBoost']
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_dist, n_iter=20,
        scoring=rmse_scorer, cv=kf, n_jobs=-1, random_state=SEED, verbose=1
    )

    xgb_search.fit(X_train, y_train)
    models['XGBoost_tuned'] = xgb_search.best_estimator_
    model_scores['XGBoost_tuned'] = -xgb_search.best_score_

    logger.info(f"XGBoost 最佳参数: {xgb_search.best_params_}")
    logger.info(f"XGBoost 最佳分数: {-xgb_search.best_score_:.4f}")

# 更新模型列表
tuned_models = {name: model for name, model in models.items()
                if name in model_scores and model_scores[name] != float('inf')}

# --------------------------
# 5. 模型集成
# --------------------------
logger.info("\n构建集成模型...")

# 5.1 加权集成
# 基于模型性能计算权重（RMSE越低，权重越高）
weights = {}
for name, score in model_scores.items():
    if score != float('inf'):
        # 使用性能的倒数作为权重基础
        weights[name] = 1 / score

# 创建加权集成模型
weighted_ensemble = WeightedEnsemble(models=tuned_models, weights=weights)

# 评估加权集成
logger.info("评估加权集成模型...")
ensemble_scores = cross_val_score(
    weighted_ensemble, X_train, y_train,
    scoring=rmse_scorer, cv=kf, n_jobs=1  # 注意：集成模型不能并行
)
ensemble_rmse = -ensemble_scores.mean()
ensemble_std = ensemble_scores.std()
logger.info(f"加权集成模型: RMSE = {ensemble_rmse:.4f} (±{ensemble_std:.4f})")

# 5.2 堆叠集成
logger.info("构建堆叠集成模型...")

# 选择表现最好的几个模型进行堆叠
top_models = [(name, tuned_models[name]) for name, _ in
              sorted(model_scores.items(), key=lambda x: x[1])[:5]]

stacking_model = StackingRegressor(
    estimators=top_models,
    final_estimator=Ridge(alpha=10.0, random_state=SEED),
    cv=kf,
    n_jobs=-1
)

# 评估堆叠集成
stacking_scores = cross_val_score(
    stacking_model, X_train, y_train,
    scoring=rmse_scorer, cv=kf, n_jobs=-1
)
stacking_rmse = -stacking_scores.mean()
stacking_std = stacking_scores.std()
logger.info(f"堆叠集成模型: RMSE = {stacking_rmse:.4f} (±{stacking_std:.4f})")

# 选择最佳模型
if stacking_rmse < ensemble_rmse and stacking_rmse < min(model_scores.values()):
    final_model = stacking_model
    final_model_type = "堆叠集成"
elif ensemble_rmse < min(model_scores.values()):
    final_model = weighted_ensemble
    final_model_type = "加权集成"
else:
    best_name = min(model_scores.items(), key=lambda x: x[1])[0]
    final_model = tuned_models[best_name]
    final_model_type = f"最佳单模型 ({best_name})"

logger.info(f"选择最终模型: {final_model_type}")

# --------------------------
# 6. 最终模型训练与保存
# --------------------------
logger.info("使用全量数据训练最终模型...")
start_time = time.time()
final_model.fit(X_train, y_train)
logger.info(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")

# 创建输出目录
os.makedirs('predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 保存模型
model_path = 'models/final_model_1.pkl'
joblib.dump(final_model, model_path)
logger.info(f"最终模型已保存至 {model_path}")

# --------------------------
# 7. 预测与结果生成
# --------------------------
logger.info("生成测试集预测...")

# 确保测试集特征与训练集一致
feature_selector = FeatureSelector(feature_names)
X_test_aligned = feature_selector.fit_transform(X_test)

# 生成预测
test_predictions = np.expm1(final_model.predict(X_test_aligned))

# 创建提交文件
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

# 确保ID排序正确
submission = submission.sort_values('Id')

# 保存结果
submission_path = 'predictions/submission.csv'
submission.to_csv(submission_path, index=False)
logger.info(f"预测结果已保存至 {submission_path}")

# --------------------------
# 8. 性能总结
# --------------------------
logger.info("\n" + "=" * 50)
logger.info("模型性能总结:")
logger.info("=" * 50)

for name, score in sorted(model_scores.items(), key=lambda x: x[1]):
    if score != float('inf'):
        logger.info(f"{name:20s}: {score:.4f} (±{model_stds.get(name, 0):.4f})")

logger.info(f"{'加权集成':20s}: {ensemble_rmse:.4f} (±{ensemble_std:.4f})")
logger.info(f"{'堆叠集成':20s}: {stacking_rmse:.4f} (±{stacking_std:.4f})")
logger.info(f"最终选择模型: {final_model_type}")

# 显示提交文件统计信息
logger.info(f"\n预测价格统计:")
logger.info(f"最小值: {submission.SalePrice.min():.2f}")
logger.info(f"最大值: {submission.SalePrice.max():.2f}")
logger.info(f"平均值: {submission.SalePrice.mean(): .2f}")
logger.info(f"中位数: {submission.SalePrice.median():.2f}")

logger.info("\n项目完成！")

# --------------------------
# 9. 数据可视化
# --------------------------
logger.info("开始数据可视化...")

try:
    # 加载原始数据用于可视化
    raw_train_df = pd.read_csv('data/train.csv')
    raw_test_df = pd.read_csv('data/test.csv')

    # 创建可视化实例
    visualizer = HousingDataVisualizer(
        train_path='data/train.csv',
        test_path='data/test.csv',
        model_path='models/final_model_1.pkl'
    )

    # 创建完整的数据可视化仪表板
    # 注意：需要将数据转换为原始尺度进行可视化
    y_train_original = np.log1p(train_processed['SalePrice'])  # 如果之前做了对数转换

    visualizer.create_dashboard(
        X_train=X_train,
        y_train=y_train_original,
        X_test=X_test_aligned,
        y_test=None  # 测试集真实值不可用
    )

except Exception as e:
    logger.error(f"数据可视化过程中出错: {e}")
