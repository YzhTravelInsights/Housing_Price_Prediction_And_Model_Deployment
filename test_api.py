#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    : 
# @File    : test_api.py
# @Version：V 0.1
# @desc :
import requests
import json
import pandas as pd

# API基础URL
BASE_URL = "http://localhost:5000"


def test_health():
    """测试API健康状态"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("健康检查:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def test_features():
    """获取特征列表"""
    try:
        response = requests.get(f"{BASE_URL}/features")
        print("特征列表:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"获取特征列表失败: {e}")
        return False


def test_prediction():
    """测试预测功能"""
    # 示例房屋数据（基于训练数据的特征）
    sample_house = {
        "GrLivArea": 1500,
        "OverallQual": 6,
        "GarageCars": 2,
        "GarageArea": 480,
        "TotalBsmtSF": 1000,
        "1stFlrSF": 1000,
        "FullBath": 2,
        "TotRmsAbvGrd": 6,
        "YearBuilt": 2000,
        "YearRemodAdd": 2000,
        "GarageYrBlt": 2000,
        "MasVnrArea": 200,
        "Fireplaces": 1,
        "BsmtFinSF1": 400,
        "LotFrontage": 60,
        "WoodDeckSF": 100,
        "2ndFlrSF": 500,
        "OpenPorchSF": 50
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_house,
            headers={"Content-Type": "application/json"}
        )

        result = response.json()
        print("预测结果:", result)
        return response.status_code == 200
    except Exception as e:
        print(f"预测请求失败: {e}")
        return False


if __name__ == "__main__":
    print("测试房价预测API...")

    # 测试健康状态
    if not test_health():
        print("API可能未启动，请先运行 app.py")
        exit()

    # 测试特征列表
    test_features()

    # 测试预测
    test_prediction()