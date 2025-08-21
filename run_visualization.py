#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    : 
# @File    : run_visualization.py
# @Version：V 0.1
# @desc :
import pandas as pd
import numpy as np
from visualization import HousingDataVisualizer

# 加载处理后的数据
train_processed = pd.read_csv('processed_data/train_processed.csv')
test_processed = pd.read_csv('processed_data/test_processed.csv')

# 准备特征和目标变量
X_train = train_processed.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
y_train = np.log1p(train_processed['SalePrice'])  # 对数转换后的目标变量

# 处理测试集
if 'SalePrice' in test_processed.columns:
    X_test = test_processed.drop(['Id', 'SalePrice'], axis=1)
else:
    X_test = test_processed.drop('Id', axis=1)

# 创建可视化实例
visualizer = HousingDataVisualizer(
    train_path='data/train.csv',      # 原始数据路径
    test_path='data/test.csv',        # 原始数据路径
    model_path='models/final_model.pkl'
)

# 运行完整可视化
visualizer.create_dashboard(X_train, y_train, X_test)