#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    :
# @File    : feature_engineering.py
# @Version：V 0.1
# @desc :
import pandas as pd
import numpy as np
from scipy.stats import skew
import os


def feature_engineering(df, is_train=True, train_stats=None):
    """
    对房价数据集进行特征工程处理，确保训练集和测试集处理逻辑一致

    参数:
        df: 待处理的DataFrame
        is_train: 是否为训练集（用于计算/使用统计量）
        train_stats: 训练集的统计量字典（仅测试集使用）

    返回:
        处理后的DataFrame和训练集统计量（如果是训练集）
    """
    # 初始化训练集统计量字典
    if is_train:
        train_stats = {}

    # --------------------------
    # 1. 缺失值处理
    # --------------------------

    # 类别型特征：缺失表示"不存在"，填充为'None'
    cat_na_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in cat_na_none:
        df[col] = df[col].fillna('None')

    # 数值型特征：缺失表示"无"，填充为0
    num_na_zero = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars',
        'GarageArea', 'PoolArea', 'MiscVal'
    ]
    for col in num_na_zero:
        df[col] = df[col].fillna(0)

    # 特殊处理：LotFrontage（临街宽度）- 用邻里的中位数填充（空间相关性）
    if is_train:
        # 计算每个邻里的中位数并保存
        train_stats['LotFrontage_median'] = df.groupby('Neighborhood')['LotFrontage'].median()
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # 仍有缺失（部分邻里无数据），用整体中位数填充
        train_stats['LotFrontage_overall_median'] = df['LotFrontage'].median()
        df['LotFrontage'] = df['LotFrontage'].fillna(train_stats['LotFrontage_overall_median'])
    else:
        # 测试集用训练集的邻里中位数填充
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(train_stats['LotFrontage_median'].get(x.name, train_stats['LotFrontage_overall_median']))
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(train_stats['LotFrontage_overall_median'])

    # 特殊处理：GarageYrBlt（车库建造年份）- 缺失用房屋建造年份填充
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # 其他低缺失率类别特征 - 用众数填充
    cat_na_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Electrical']
    for col in cat_na_mode:
        if is_train:
            train_stats[f'{col}_mode'] = df[col].mode()[0]
            df[col] = df[col].fillna(train_stats[f'{col}_mode'])
        else:
            df[col] = df[col].fillna(train_stats[f'{col}_mode'])

    # --------------------------
    # 2. 特征创建（核心价值）
    # --------------------------

    # 总居住面积（地上+地下）
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']

    # 总浴室数量（全浴室算1，半浴室算0.5）
    df['TotalBath'] = (
            df['FullBath'] + 0.5 * df['HalfBath'] +
            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )

    # 房屋年龄（销售时）
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # 装修后年限（销售时）
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # 总门廊/露台面积
    df['TotalPorchSF'] = (
            df['OpenPorchSF'] + df['EnclosedPorch'] +
            df['3SsnPorch'] + df['ScreenPorch']
    )

    # 房屋质量与面积的交互特征（质量高且面积大的房子更贵）
    df['QualAreaInteract'] = df['OverallQual'] * df['GrLivArea']

    # 车库与房屋建造时间差（同步建造更值钱）
    df['GarageHouseDiff'] = df['GarageYrBlt'] - df['YearBuilt']
    # 修正：车库不可能在房屋建造前建造，负值设为0
    df['GarageHouseDiff'] = df['GarageHouseDiff'].clip(lower=0)

    # 房屋总房间数
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['BedroomAbvGr']

    # 每平方米价格（用于后续分析，非直接特征）
    if 'SalePrice' in df.columns:
        df['PricePerSF'] = df['SalePrice'] / df['TotalSF'].replace(0, 1)  # 避免除零

    # --------------------------
    # 3. 偏态特征修正
    # --------------------------

    # 筛选数值特征（排除ID和目标变量）
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    numeric_feats = [f for f in numeric_feats if f not in ['Id', 'SalePrice', 'PricePerSF']]

    if is_train:
        # 计算训练集偏度并保存高偏度特征
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        train_stats['high_skew_feats'] = skewed_feats[abs(skewed_feats) > 0.75].index.tolist()
        print(f"对 {len(train_stats['high_skew_feats'])} 个高偏度特征进行对数转换")

    # 对高偏度特征应用log1p转换（添加预处理避免无效值）
    for feat in train_stats['high_skew_feats']:
        # 确保所有值都大于-1，避免log1p计算错误
        df[feat] = df[feat].clip(lower=-0.5)  # 确保1 + x >= 0.5
        df[feat] = np.log1p(df[feat])

    # --------------------------
    # 4. 类别变量编码
    # --------------------------

    # 独热编码（自动处理所有类别特征）
    df = pd.get_dummies(df, drop_first=True)  # drop_first减少多重共线性

    # --------------------------
    # 5. 清理不需要的特征
    # --------------------------

    # 移除原始面积特征（已被总特征替代）
    drop_feats = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GrLivArea',
                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                  'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                  'TotRmsAbvGrd', 'BedroomAbvGr']
    df = df.drop(columns=drop_feats, errors='ignore')

    if is_train:
        return df, train_stats
    else:
        return df


# --------------------------
# 主函数：处理数据并保存结果
# --------------------------
if __name__ == "__main__":
    # 创建保存处理后数据的目录
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据（使用copy()不修改原始数据）
    try:
        train_df = pd.read_csv('data/train.csv').copy()
        test_df = pd.read_csv('data/test.csv').copy()
        print("原始数据加载成功")
    except FileNotFoundError:
        print("错误：未找到数据文件，请检查路径是否正确")
        exit()
    except Exception as e:
        print(f"加载数据时发生错误：{str(e)}")
        exit()

    # 处理训练集（原始数据不会被修改）
    try:
        train_processed, train_stats = feature_engineering(train_df, is_train=True)
        print(f"训练集处理后形状: {train_processed.shape}")
    except Exception as e:
        print(f"处理训练集时发生错误：{str(e)}")
        exit()

    # 处理测试集（使用训练集统计量）
    try:
        test_processed = feature_engineering(test_df, is_train=False, train_stats=train_stats)
        print(f"测试集处理后形状: {test_processed.shape}")
    except Exception as e:
        print(f"处理测试集时发生错误：{str(e)}")
        exit()

    # 确保训练集和测试集特征对齐
    try:
        final_train, final_test = train_processed.align(
            test_processed,
            join='left',  # 保留训练集所有特征
            axis=1,
            fill_value=0  # 测试集缺失特征用0填充
        )
    except Exception as e:
        print(f"特征对齐时发生错误：{str(e)}")
        exit()

    # 保存处理后的数据集到新文件
    try:
        train_save_path = os.path.join(output_dir, 'train_processed.csv')
        test_save_path = os.path.join(output_dir, 'test_processed.csv')

        final_train.to_csv(train_save_path, index=False)
        final_test.to_csv(test_save_path, index=False)

        print(f"处理后的训练集已保存至: {train_save_path}")
        print(f"处理后的测试集已保存至: {test_save_path}")
        print(f"原始数据文件未被修改")
    except Exception as e:
        print(f"保存文件时发生错误：{str(e)}")
        exit()
