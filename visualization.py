#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    : 
# @File    : visualization.py
# @Version：V 0.1
# @desc :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class HousingDataVisualizer:
    def __init__(self, train_path, test_path, model_path=None):
        """
        初始化可视化工具

        参数:
            train_path: 训练数据路径
            test_path: 测试数据路径
            model_path: 模型路径(可选)
        """
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.model = joblib.load(model_path) if model_path else None
        self.predictions = None

        # 创建输出目录
        os.makedirs('visualizations', exist_ok=True)

    def plot_target_distribution(self):
        """绘制目标变量分布"""
        plt.figure(figsize=(15, 6))

        # 原始价格分布
        plt.subplot(1, 2, 1)
        sns.histplot(self.train_df['SalePrice'], kde=True)
        plt.title('原始房价分布')
        plt.xlabel('价格')
        plt.ylabel('频数')

        # 对数转换后的价格分布
        plt.subplot(1, 2, 2)
        sns.histplot(np.log1p(self.train_df['SalePrice']), kde=True)
        plt.title('对数转换后房价分布')
        plt.xlabel('对数价格')
        plt.ylabel('频数')

        plt.tight_layout()
        plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_missing_values(self):
        """绘制缺失值热力图"""
        # 计算缺失值比例
        missing_data = pd.DataFrame({
            '缺失比例': self.train_df.isnull().sum() / len(self.train_df) * 100
        }).sort_values('缺失比例', ascending=False)

        # 只显示有缺失值的特征
        missing_data = missing_data[missing_data['缺失比例'] > 0]

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.train_df[missing_data.index].isnull(),
                    yticklabels=False, cbar=False, cmap='viridis')
        plt.title('缺失值热力图')
        plt.savefig('visualizations/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制缺失值比例条形图
        plt.figure(figsize=(12, 8))
        missing_data = missing_data.head(20)  # 只显示前20个
        sns.barplot(x=missing_data.index, y=missing_data['缺失比例'])
        plt.title('缺失值比例最高的20个特征')
        plt.xticks(rotation=90)
        plt.ylabel('缺失比例 (%)')
        plt.savefig('visualizations/missing_values_barplot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, top_n=20):
        """绘制相关性矩阵热力图"""
        # 计算数值型特征的相关性
        numeric_features = self.train_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_features.corr()

        # 获取与目标变量最相关的特征
        corr_with_target = corr_matrix['SalePrice'].sort_values(ascending=False)
        top_features = corr_with_target[1:top_n + 1].index  # 排除目标变量本身

        # 绘制热力图
        plt.figure(figsize=(14, 12))
        sns.heatmap(numeric_features[top_features].corr(),
                    annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, square=True)
        plt.title(f'房价与最相关{top_n}个特征的相关性矩阵')
        plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制与目标变量最相关的特征条形图
        plt.figure(figsize=(12, 8))
        corr_with_target = corr_with_target[1:top_n + 1]  # 排除目标变量本身
        sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
        plt.title(f'与房价最相关的前{top_n}个特征')
        plt.xlabel('相关系数')
        plt.savefig('visualizations/target_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_vs_target(self, feature_names):
        """绘制特征与目标变量的关系"""
        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        plt.figure(figsize=(18, 6 * n_rows))

        for i, feature in enumerate(feature_names):
            plt.subplot(n_rows, n_cols, i + 1)

            if self.train_df[feature].dtype == 'object' or len(self.train_df[feature].unique()) < 10:
                # 分类特征或离散数值特征
                sns.boxplot(x=feature, y='SalePrice', data=self.train_df)
                plt.xticks(rotation=45)
            else:
                # 连续数值特征
                sns.scatterplot(x=feature, y='SalePrice', data=self.train_df, alpha=0.6)

            plt.title(f'{feature} vs SalePrice')

        plt.tight_layout()
        plt.savefig('visualizations/feature_vs_target.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_performance(self, X_train, y_train, X_test, y_test=None):
        """绘制模型性能评估图"""
        if not self.model:
            print("未加载模型，无法评估性能")
            return

        # 生成预测
        train_pred = self.model.predict(X_train)
        self.predictions = self.model.predict(X_test)

        # 如果提供了测试集真实值，计算测试集性能
        if y_test is not None:
            test_rmse = np.sqrt(mean_squared_error(y_test, self.predictions))
            test_r2 = r2_score(y_test, self.predictions)
            print(f"测试集 RMSE: {test_rmse:.4f}")
            print(f"测试集 R²: {test_r2:.4f}")

        # 计算训练集性能
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"训练集 R²: {train_r2:.4f}")

        # 绘制预测值与真实值散点图
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_train, train_pred, alpha=0.5)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'训练集预测 vs 真实值 (R² = {train_r2:.4f})')

        # 绘制残差图
        plt.subplot(1, 2, 2)
        residuals = y_train - train_pred
        plt.scatter(train_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('训练集残差图')

        plt.tight_layout()
        plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性图"""
        if not self.model:
            print("未加载模型，无法获取特征重要性")
            return

        # 获取特征重要性（不同模型有不同的方法）
        try:
            if hasattr(self.model, 'feature_importances_'):
                # 树模型
                importances = self.model.feature_importances_
                feature_names = self.model.feature_names_in_
            elif hasattr(self.model, 'coef_'):
                # 线性模型
                importances = np.abs(self.model.coef_)
                if hasattr(self.model, 'feature_names_in_'):
                    feature_names = self.model.feature_names_in_
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
            elif hasattr(self.model, 'estimators_'):
                # 集成模型
                importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
                feature_names = self.model.feature_names_in_
            else:
                print("无法提取特征重要性")
                return

            # 创建特征重要性DataFrame
            feat_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            # 绘制条形图
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feat_imp)
            plt.title(f'前{top_n}个最重要特征')
            plt.xlabel('重要性')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"提取特征重要性时出错: {e}")

    def create_dashboard(self, X_train, y_train, X_test, y_test=None):
        """创建完整的数据可视化仪表板"""
        print("开始创建数据可视化仪表板...")

        # 1. 目标变量分布
        print("绘制目标变量分布...")
        self.plot_target_distribution()

        # 2. 缺失值分析
        print("分析缺失值...")
        self.plot_missing_values()

        # 3. 相关性分析
        print("分析特征相关性...")
        self.plot_correlation_matrix(top_n=15)

        # 4. 重要特征与目标变量关系
        print("分析重要特征与目标变量关系...")
        numeric_features = self.train_df.select_dtypes(include=[np.number])
        corr_with_target = numeric_features.corr()['SalePrice'].sort_values(ascending=False)
        top_features = corr_with_target[1:7].index  # 选择前6个最相关的特征
        self.plot_feature_vs_target(top_features)

        # 5. 模型性能评估
        print("评估模型性能...")
        self.plot_model_performance(X_train, y_train, X_test, y_test)

        # 6. 特征重要性
        print("分析特征重要性...")
        self.plot_feature_importance(top_n=15)

        print("数据可视化完成！所有图表已保存到 'visualizations' 文件夹")