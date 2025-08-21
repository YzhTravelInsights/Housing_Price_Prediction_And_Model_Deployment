#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/21
# @Author  : yzh
# @Site    : 
# @File    : app.py
# @Version：V 0.1
# @desc :
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型和特征信息
model = None
feature_names = None
scaler = None


def load_model():
    """加载模型和特征信息"""
    global model, feature_names

    try:
        # 加载模型
        model_path = os.path.join('models', 'final_model0.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model = joblib.load(model_path)
        logger.info("模型加载成功")

        # 加载特征名称
        feature_path = os.path.join('models', 'feature_names.pkl')
        if os.path.exists(feature_path):
            feature_names = joblib.load(feature_path)
            logger.info(f"特征名称加载成功，共 {len(feature_names)} 个特征")
        else:
            logger.warning("特征名称文件未找到，将使用默认特征处理")

    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        raise


def prepare_features(input_data):
    """准备特征数据以供模型预测"""
    try:
        # 将输入数据转换为DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)

        # 确保所有特征都存在
        if feature_names is not None:
            # 添加缺失的特征并填充默认值
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # 选择正确的特征顺序
            input_df = input_df[feature_names]

        return input_df

    except Exception as e:
        logger.error(f"准备特征时出错: {e}")
        raise


@app.route('/')
def home():
    """首页路由"""
    return jsonify({
        'message': '房价预测API服务',
        'status': '运行中',
        'endpoints': {
            '预测房价': '/predict (POST)',
            'API状态': '/health (GET)'
        }
    })


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """房价预测端点"""
    try:
        # 获取输入数据
        data = request.get_json()

        if not data:
            return jsonify({'error': '未提供数据'}), 400

        logger.info(f"收到预测请求: {data}")

        # 准备特征数据
        features = prepare_features(data)

        # 进行预测
        prediction_log = model.predict(features)

        # 转换回原始价格尺度
        prediction = np.expm1(prediction_log)[0]

        logger.info(f"预测完成: {prediction:.2f}")

        return jsonify({
            'predicted_price': round(prediction, 2),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def get_features():
    """获取模型使用的特征列表"""
    if feature_names is None:
        return jsonify({'error': '特征信息不可用'}), 500

    return jsonify({
        'features': feature_names,
        'count': len(feature_names)
    })


# 应用启动时加载模型
if __name__ == '__main__':
    logger.info("正在启动房价预测API服务...")

    try:
        load_model()
        logger.info("API服务启动成功")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"启动失败: {e}")