#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, log_loss, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.base import clone, BaseEstimator
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer

# 添加贝叶斯优化所需库
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from functools import partial
from skopt.utils import use_named_args

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import torchviz
from torchviz import make_dot
from tqdm import tqdm
import xgboost as xgb
from lightgbm import LGBMClassifier
import seaborn as sns
from itertools import cycle
import pickle
import logging
import copy
from collections import defaultdict
import matplotlib.gridspec as gridspec

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##########################################
# 1. 数据加载与预处理
##########################################
# 修改数据文件路径为实际路径
data = pd.read_excel(r'F:\Program_Database\paperDatabase2\02_python_lib\pythonProject\Mei_CNN_RF_SVM_ANN\EN中国蔬菜镉含量数据库_all9.xlsx')

# 新目标变量转换函数----将THQ映射为风险等级：0（无风险）、1（低风险）、2（中风险）、3（高风险）
def map_thq_to_risk(thq):
    if thq <= 0.5:
        return 0
    elif thq <= 1:
        return 1
    elif thq <= 2:
        return 2
    else:
        return 3

# 分别计算城市和农村中男性和女性的风险等级（直接转换为数值型目标）
data['Urban Risk (Male)']   = data['Urban THQ (Male)'].apply(map_thq_to_risk)
data['Urban Risk (Female)'] = data['Urban THQ (Female)'].apply(map_thq_to_risk)
data['Rural Risk (Male)']   = data['Rural THQ (Male)'].apply(map_thq_to_risk)
data['Rural Risk (Female)'] = data['Rural THQ (Female)'].apply(map_thq_to_risk)

print("各风险标签类别分布：")
print("Urban Risk (Male):")
print(data['Urban Risk (Male)'].value_counts())
print("Urban Risk (Female):")
print(data['Urban Risk (Female)'].value_counts())
print("Rural Risk (Male):")
print(data['Rural Risk (Male)'].value_counts())
print("Rural Risk (Female):")
print(data['Rural Risk (Female)'].value_counts())

# 清洗列名
logging.info("清洗列名...")
data.columns = data.columns.str.strip().str.replace('\xa0', ' ', regex=False)
logging.info(f"清洗后的列名：\n{data.columns.tolist()}")

##########################################
# 2. 对象类型列编码
##########################################
# 预定义分类列（请根据数据实际情况调整）
categorical_columns = ['Region', 'Province', 'Climate Zone',
                       'Major Veg Category', 'Specific Veg Type', 'Season']

object_columns = data.select_dtypes(include=['object']).columns
logging.info(f"对象类型的列：\n{object_columns.tolist()}")

category_mappings = {}
for column in object_columns:
    logging.info(f"处理列：{column}")
    le = LabelEncoder()
    data[column] = data[column].fillna('Missing')
    try:
        data[column] = le.fit_transform(data[column].astype(str))
        data[column] = data[column].astype('float64')
        category_mappings[column] = le
        logging.info(f"成功编码列：{column}")
        if column in categorical_columns:
            logging.info(f"预定义分类列 {column} 已编码")
    except Exception as e:
        logging.error(f"编码列 {column} 时出错：{e}")
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].fillna(data[column].mean())
        data[column] = data[column].astype('float64')
        logging.info(f"列 {column} 已转换为数值类型")

##########################################
# 3. 构造特征集：分城市男性、城市女性、农村男性、农村女性建模
##########################################
# 定义共有特征、各组特有特征
common_features = [
    'Region', 'Province', 'Climate Zone', 'Major Veg Category', 'Specific Veg Type',
    'Soil Cadmium (mg/kg)', 'BCF', 'Season', 'Year',
    'Vegetable Cadmium (mg/kg)', 'SCC', 'pH', 'SOM', 'CEC'
]
urban_male_specific_features = ['Urban Veg Consumption (kg/year/capita)', 'Urban Body Weight (Male)']
urban_female_specific_features = ['Urban Veg Consumption (kg/year/capita)', 'Urban Body Weight (Female)']
rural_male_specific_features = ['Rural Veg Consumption (kg/year/capita)', 'Rural Body Weight (Male)']
rural_female_specific_features = ['Rural Veg Consumption (kg/year/capita)', 'Rural Body Weight (Female)']

# 构造四个特征数据集
X_urban_male = data[common_features + urban_male_specific_features].copy()
X_urban_female = data[common_features + urban_female_specific_features].copy()
X_rural_male = data[common_features + rural_male_specific_features].copy()
X_rural_female = data[common_features + rural_female_specific_features].copy()

##########################################
# 4. 构造树模型与非树模型数据集（分别处理四个组）
##########################################
# 树模型直接使用原始数据（保留NaN信息）
X_tree_urban_male = X_urban_male.copy()
X_tree_urban_female = X_urban_female.copy()
X_tree_rural_male = X_rural_male.copy()
X_tree_rural_female = X_rural_female.copy()

# 非树模型：添加缺失指标、填充特殊值，再用 StandardScaler 标准化
X_non_tree_urban_male = X_urban_male.copy()
X_non_tree_urban_female = X_urban_female.copy()
X_non_tree_rural_male = X_rural_male.copy()
X_non_tree_rural_female = X_rural_female.copy()

special_value = -999
# 处理城市男性数据
for col in X_non_tree_urban_male.select_dtypes(include=['float64', 'int64']).columns:
    X_non_tree_urban_male[f'{col}_missing'] = X_non_tree_urban_male[col].isnull().astype(int)
    X_non_tree_urban_male[col] = X_non_tree_urban_male[col].fillna(special_value)

# 处理城市女性数据
for col in X_non_tree_urban_female.select_dtypes(include=['float64', 'int64']).columns:
    X_non_tree_urban_female[f'{col}_missing'] = X_non_tree_urban_female[col].isnull().astype(int)
    X_non_tree_urban_female[col] = X_non_tree_urban_female[col].fillna(special_value)

# 处理农村男性数据
for col in X_non_tree_rural_male.select_dtypes(include=['float64', 'int64']).columns:
    X_non_tree_rural_male[f'{col}_missing'] = X_non_tree_rural_male[col].isnull().astype(int)
    X_non_tree_rural_male[col] = X_non_tree_rural_male[col].fillna(special_value)

# 处理农村女性数据
for col in X_non_tree_rural_female.select_dtypes(include=['float64', 'int64']).columns:
    X_non_tree_rural_female[f'{col}_missing'] = X_non_tree_rural_female[col].isnull().astype(int)
    X_non_tree_rural_female[col] = X_non_tree_rural_female[col].fillna(special_value)

# 标准化数据
scaler_non_tree_urban_male = StandardScaler()
X_non_tree_urban_male_scaled = pd.DataFrame(scaler_non_tree_urban_male.fit_transform(X_non_tree_urban_male),
                                        columns=X_non_tree_urban_male.columns)

scaler_non_tree_urban_female = StandardScaler()
X_non_tree_urban_female_scaled = pd.DataFrame(scaler_non_tree_urban_female.fit_transform(X_non_tree_urban_female),
                                        columns=X_non_tree_urban_female.columns)

scaler_non_tree_rural_male = StandardScaler()
X_non_tree_rural_male_scaled = pd.DataFrame(scaler_non_tree_rural_male.fit_transform(X_non_tree_rural_male),
                                         columns=X_non_tree_rural_male.columns)

scaler_non_tree_rural_female = StandardScaler()
X_non_tree_rural_female_scaled = pd.DataFrame(scaler_non_tree_rural_female.fit_transform(X_non_tree_rural_female),
                                         columns=X_non_tree_rural_female.columns)

##########################################
# 5. 构造目标变量（分别取出四个风险标签）
##########################################
Y_urban_male = data['Urban Risk (Male)'].values
Y_urban_female = data['Urban Risk (Female)'].values
Y_rural_male = data['Rural Risk (Male)'].values
Y_rural_female = data['Rural Risk (Female)'].values

##########################################
# 6. 数据集划分（四个组分别划分）
##########################################
# 城市男性----树模型数据集
X_tree_urban_male_train_val, X_tree_urban_male_test, Y_urban_male_train_val, Y_urban_male_test = train_test_split(X_tree_urban_male, Y_urban_male, test_size=0.15, random_state=42)
X_tree_urban_male_train, X_tree_urban_male_val, Y_urban_male_train, Y_urban_male_val = train_test_split(X_tree_urban_male_train_val, Y_urban_male_train_val, test_size=0.1765, random_state=42)

# 城市男性----非树模型数据集
X_non_tree_urban_male_train_val, X_non_tree_urban_male_test, Y_urban_male_train_val, Y_urban_male_test = train_test_split(X_non_tree_urban_male_scaled, Y_urban_male, test_size=0.15, random_state=42)
X_non_tree_urban_male_train, X_non_tree_urban_male_val, Y_urban_male_train, Y_urban_male_val = train_test_split(X_non_tree_urban_male_train_val, Y_urban_male_train_val, test_size=0.1765, random_state=42)

# 城市女性----树模型数据集
X_tree_urban_female_train_val, X_tree_urban_female_test, Y_urban_female_train_val, Y_urban_female_test = train_test_split(X_tree_urban_female, Y_urban_female, test_size=0.15, random_state=42)
X_tree_urban_female_train, X_tree_urban_female_val, Y_urban_female_train, Y_urban_female_val = train_test_split(X_tree_urban_female_train_val, Y_urban_female_train_val, test_size=0.1765, random_state=42)

# 城市女性----非树模型数据集
X_non_tree_urban_female_train_val, X_non_tree_urban_female_test, Y_urban_female_train_val, Y_urban_female_test = train_test_split(X_non_tree_urban_female_scaled, Y_urban_female, test_size=0.15, random_state=42)
X_non_tree_urban_female_train, X_non_tree_urban_female_val, Y_urban_female_train, Y_urban_female_val = train_test_split(X_non_tree_urban_female_train_val, Y_urban_female_train_val, test_size=0.1765, random_state=42)

# 农村男性----树模型数据集
X_tree_rural_male_train_val, X_tree_rural_male_test, Y_rural_male_train_val, Y_rural_male_test = train_test_split(X_tree_rural_male, Y_rural_male, test_size=0.15, random_state=42)
X_tree_rural_male_train, X_tree_rural_male_val, Y_rural_male_train, Y_rural_male_val = train_test_split(X_tree_rural_male_train_val, Y_rural_male_train_val, test_size=0.1765, random_state=42)

# 农村男性----非树模型数据集
X_non_tree_rural_male_train_val, X_non_tree_rural_male_test, Y_rural_male_train_val, Y_rural_male_test = train_test_split(X_non_tree_rural_male_scaled, Y_rural_male, test_size=0.15, random_state=42)
X_non_tree_rural_male_train, X_non_tree_rural_male_val, Y_rural_male_train, Y_rural_male_val = train_test_split(X_non_tree_rural_male_train_val, Y_rural_male_train_val, test_size=0.1765, random_state=42)

# 农村女性----树模型数据集
X_tree_rural_female_train_val, X_tree_rural_female_test, Y_rural_female_train_val, Y_rural_female_test = train_test_split(X_tree_rural_female, Y_rural_female, test_size=0.15, random_state=42)
X_tree_rural_female_train, X_tree_rural_female_val, Y_rural_female_train, Y_rural_female_val = train_test_split(X_tree_rural_female_train_val, Y_rural_female_train_val, test_size=0.1765, random_state=42)

# 农村女性----非树模型数据集
X_non_tree_rural_female_train_val, X_non_tree_rural_female_test, Y_rural_female_train_val, Y_rural_female_test = train_test_split(X_non_tree_rural_female_scaled, Y_rural_female, test_size=0.15, random_state=42)
X_non_tree_rural_female_train, X_non_tree_rural_female_val, Y_rural_female_train, Y_rural_female_val = train_test_split(X_non_tree_rural_female_train_val, Y_rural_female_train_val, test_size=0.1765, random_state=42)

# 对于不能处理 NaN 的模型，对树模型数据集进行均值填充（分别处理四个组）
imputer_tree_urban_male = SimpleImputer(missing_values=np.nan, strategy='mean')
X_tree_urban_male_train_imputed = pd.DataFrame(imputer_tree_urban_male.fit_transform(X_tree_urban_male_train), columns=X_tree_urban_male_train.columns, index=X_tree_urban_male_train.index)
X_tree_urban_male_val_imputed = pd.DataFrame(imputer_tree_urban_male.transform(X_tree_urban_male_val), columns=X_tree_urban_male_val.columns, index=X_tree_urban_male_val.index)
X_tree_urban_male_test_imputed = pd.DataFrame(imputer_tree_urban_male.transform(X_tree_urban_male_test), columns=X_tree_urban_male_test.columns, index=X_tree_urban_male_test.index)

imputer_tree_urban_female = SimpleImputer(missing_values=np.nan, strategy='mean')
X_tree_urban_female_train_imputed = pd.DataFrame(imputer_tree_urban_female.fit_transform(X_tree_urban_female_train), columns=X_tree_urban_female_train.columns, index=X_tree_urban_female_train.index)
X_tree_urban_female_val_imputed = pd.DataFrame(imputer_tree_urban_female.transform(X_tree_urban_female_val), columns=X_tree_urban_female_val.columns, index=X_tree_urban_female_val.index)
X_tree_urban_female_test_imputed = pd.DataFrame(imputer_tree_urban_female.transform(X_tree_urban_female_test), columns=X_tree_urban_female_test.columns, index=X_tree_urban_female_test.index)

imputer_tree_rural_male = SimpleImputer(missing_values=np.nan, strategy='mean')
X_tree_rural_male_train_imputed = pd.DataFrame(imputer_tree_rural_male.fit_transform(X_tree_rural_male_train), columns=X_tree_rural_male_train.columns, index=X_tree_rural_male_train.index)
X_tree_rural_male_val_imputed = pd.DataFrame(imputer_tree_rural_male.transform(X_tree_rural_male_val), columns=X_tree_rural_male_val.columns, index=X_tree_rural_male_val.index)
X_tree_rural_male_test_imputed = pd.DataFrame(imputer_tree_rural_male.transform(X_tree_rural_male_test), columns=X_tree_rural_male_test.columns, index=X_tree_rural_male_test.index)

imputer_tree_rural_female = SimpleImputer(missing_values=np.nan, strategy='mean')
X_tree_rural_female_train_imputed = pd.DataFrame(imputer_tree_rural_female.fit_transform(X_tree_rural_female_train), columns=X_tree_rural_female_train.columns, index=X_tree_rural_female_train.index)
X_tree_rural_female_val_imputed = pd.DataFrame(imputer_tree_rural_female.transform(X_tree_rural_female_val), columns=X_tree_rural_female_val.columns, index=X_tree_rural_female_val.index)
X_tree_rural_female_test_imputed = pd.DataFrame(imputer_tree_rural_female.transform(X_tree_rural_female_test), columns=X_tree_rural_female_test.columns, index=X_tree_rural_female_test.index)

##########################################
# 贝叶斯优化函数定义（针对单输出分类修改）
##########################################
# 定义优化RandomForest
def optimize_rf(X_train, y_train, n_iter=10):
    """使用贝叶斯优化对RandomForestClassifier进行超参数优化"""
    logging.info("开始RandomForest贝叶斯优化...")
    
    # 定义参数空间
    param_space = [
        Integer(50, 300, name='n_estimators'),
        Integer(5, 30, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Categorical(['sqrt', 'log2', None], name='max_features')
    ]
    
    # 定义优化目标函数
    @use_named_args(param_space)
    def objective(**params):
        rf = RandomForestClassifier(random_state=42, **params)
        rf.fit(X_train, y_train)
        return -accuracy_score(y_train, rf.predict(X_train))  # 最小化负准确率
    
    # 执行贝叶斯优化
    from skopt import gp_minimize
    result = gp_minimize(objective, param_space, n_calls=max(10, n_iter), random_state=42, verbose=True)
    
    # 获取最佳参数
    best_params = {param_space[i].name: result.x[i] for i in range(len(param_space))}
    logging.info(f"RandomForest最佳参数: {best_params}")
    
    # 使用最佳参数创建模型
    best_rf = RandomForestClassifier(random_state=42, **best_params)
    best_model = best_rf.fit(X_train, y_train)
    
    return best_model

# 优化XGBoost
def optimize_xgb(X_train, y_train, n_iter=10):
    """使用贝叶斯优化对XGBClassifier进行超参数优化"""
    logging.info("开始XGBoost贝叶斯优化...")
    
    # 定义参数空间
    param_space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, 'log-uniform', name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Real(0.5, 1.0, 'uniform', name='subsample'),
        Real(0.5, 1.0, 'uniform', name='colsample_bytree'),
        Real(0, 5, 'uniform', name='gamma'),
        Integer(1, 10, name='min_child_weight')
    ]
    
    # 定义优化目标函数
    @use_named_args(param_space)
    def objective(**params):
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=4,
            **params
        )
        xgb_model.fit(X_train, y_train)
        return -accuracy_score(y_train, xgb_model.predict(X_train))  # 最小化负准确率
    
    # 执行贝叶斯优化
    from skopt import gp_minimize
    result = gp_minimize(objective, param_space, n_calls=max(10, n_iter), random_state=42, verbose=True)
    
    # 获取最佳参数
    best_params = {param_space[i].name: result.x[i] for i in range(len(param_space))}
    logging.info(f"XGBoost最佳参数: {best_params}")
    
    # 使用最佳参数创建模型
    best_xgb = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=4,
        **best_params
    )
    best_model = best_xgb.fit(X_train, y_train)
    
    return best_model

# 优化GBDT
def optimize_gbdt(X_train, y_train, n_iter=10):
    """使用贝叶斯优化对GradientBoostingClassifier进行超参数优化"""
    logging.info("开始GBDT贝叶斯优化...")
    
    # 定义参数空间
    param_space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, 'log-uniform', name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Real(0.5, 1.0, 'uniform', name='subsample'),
        Categorical(['sqrt', 'log2', None], name='max_features')
    ]
    
    # 定义优化目标函数
    @use_named_args(param_space)
    def objective(**params):
        gbdt = GradientBoostingClassifier(random_state=42, **params)
        gbdt.fit(X_train, y_train)
        return -accuracy_score(y_train, gbdt.predict(X_train))  # 最小化负准确率
    
    # 执行贝叶斯优化
    from skopt import gp_minimize
    result = gp_minimize(objective, param_space, n_calls=max(10, n_iter), random_state=42, verbose=True)
    
    # 获取最佳参数
    best_params = {param_space[i].name: result.x[i] for i in range(len(param_space))}
    logging.info(f"GBDT最佳参数: {best_params}")
    
    # 使用最佳参数创建模型
    best_gbdt = GradientBoostingClassifier(random_state=42, **best_params)
    best_model = best_gbdt.fit(X_train, y_train)
    
    return best_model

# 优化LightGBM
def optimize_lgbm(X_train, y_train, n_iter=10):
    """使用贝叶斯优化对LGBMClassifier进行超参数优化"""
    logging.info("开始LightGBM贝叶斯优化...")
    
    # 定义参数空间
    param_space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, 'log-uniform', name='learning_rate'),
        Integer(20, 100, name='num_leaves'),
        Integer(3, 10, name='max_depth'),
        Integer(10, 50, name='min_child_samples'),
        Real(0.5, 1.0, 'uniform', name='subsample'),
        Real(0.5, 1.0, 'uniform', name='colsample_bytree')
    ]
    
    # 定义优化目标函数
    @use_named_args(param_space)
    def objective(**params):
        lgbm = LGBMClassifier(
            random_state=42,
            objective='multiclass',
            num_class=4,
            verbose=-1,
            **params
        )
        lgbm.fit(X_train, y_train)
        return -accuracy_score(y_train, lgbm.predict(X_train))  # 最小化负准确率
    
    # 执行贝叶斯优化
    from skopt import gp_minimize
    result = gp_minimize(objective, param_space, n_calls=max(10, n_iter), random_state=42, verbose=True)
    
    # 获取最佳参数
    best_params = {param_space[i].name: result.x[i] for i in range(len(param_space))}
    logging.info(f"LightGBM最佳参数: {best_params}")
    
    # 使用最佳参数创建模型
    best_lgbm = LGBMClassifier(
        random_state=42,
        objective='multiclass',
        num_class=4,
        verbose=-1,
        **best_params
    )
    best_model = best_lgbm.fit(X_train, y_train)
    
    return best_model

##########################################
# 以下为函数定义部分（保持不变）
##########################################

def evaluate_model(model, X, Y):
    """评估模型性能"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # 区分 PyTorch 模型和其他模型
    if hasattr(model, 'parameters'):  # PyTorch 模型
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        Y_tensor = torch.LongTensor(Y).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            # 获取概率输出用于 ROC AUC 计算
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
    else:  # 传统机器学习模型
        predicted = model.predict(X)
        # 获取概率输出用于 ROC AUC 计算
        probas = model.predict_proba(X)
    
    # 计算评估指标
    try:
        auc_score = roc_auc_score(Y, probas, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = -1
        print("Warning: Unable to calculate AUC score")
    
    acc = accuracy_score(Y, predicted)
    se = recall_score(Y, predicted, average='macro')
    f1 = f1_score(Y, predicted, average='macro')
    
    return {
        'AUC': auc_score,
        'ACC': acc,
        'SE': se,
        'F1': f1
    }

def save_cnn_model(model, path='cnn_model.pth'):
    torch.save(model.state_dict(), path)

def plot_cnn_convergence(train_losses, val_losses, target):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{target} CNN Convergence Curve', fontname='Times New Roman', fontsize=16)
    plt.xlabel('Epochs', fontname='Times New Roman', fontsize=15)
    plt.ylabel('Loss', fontname='Times New Roman', fontsize=15)
    plt.legend(prop={'family': 'Times New Roman', 'size': 15})
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'{target}_cnn_convergence.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    df = pd.DataFrame({
        'Epoch': range(1, len(train_losses)+1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    })
    df.to_excel(f'{target}_cnn_convergence_data.xlsx', index=False)

def plot_model_evaluation(train_metrics, val_metrics, target):
    models = ['CNN', 'RF', 'SVM', 'XGBoost', 'GBDT', 'LightGBM']
    metrics = ['AUC', 'ACC', 'SE', 'F1']
    display_names = ['AUC', 'ACC', 'SE', 'F1 score']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    
    new_colors = {
        'CNN': '#004775',
        'RF': '#41A0BC', 
        'SVM': '#E45C5E',
        'XGBoost': '#8E44AD',
        'GBDT': '#27AE60',
        'LightGBM': '#F39C12'
    }
    # 训练集图表 - 使用train_metrics
    bar_width = 0.13
    for i, model in enumerate(models):
        if train_metrics[i] is not None:
            train_values = [train_metrics[i].get(metric, 0) for metric in metrics]
            ax1.bar(np.arange(4) + i * bar_width, train_values,
                    width=bar_width, label=model, color=new_colors[model])
    ax1.set_title(f'{target} Model Performance on Training Set (10-fold CV)', fontsize=16, fontname='Times New Roman')
    ax1.set_xticks(np.arange(4) + 0.33)
    ax1.set_xticklabels(display_names, fontsize=18, fontname='Times New Roman')
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=18, prop={'family': 'Times New Roman'}, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax1.set_ylabel('Score', fontsize=18, fontname='Times New Roman')

    # 测试集图表 - 使用test_metrics
    for i, model in enumerate(models):
        if val_metrics[i] is not None:
            val_values = [val_metrics[i].get(metric, 0) for metric in metrics]
            ax2.bar(np.arange(4) + i * bar_width, val_values,
                    width=bar_width, label=model, color=new_colors[model])
    ax2.set_title(f'{target} Model Performance on Test Set (10-fold CV)', fontsize=16, fontname='Times New Roman')
    ax2.set_xticks(np.arange(4) + 0.33)
    ax2.set_xticklabels(display_names, fontsize=18, fontname='Times New Roman')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=18, prop={'family': 'Times New Roman'}, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax2.set_ylabel('Score', fontsize=18, fontname='Times New Roman')
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=18)
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
    plt.tight_layout()
    plt.savefig(f'{target}_model_evaluation.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    
    data_list = []
    for i, model in enumerate(models):
        for j, (metric, display_name) in enumerate(zip(metrics, display_names)):
            if train_metrics[i] is not None and val_metrics[i] is not None:
                data_list.append({
                    'Model': model,
                    'Metric': display_name,
                    'Train': train_metrics[i].get(metric, 0),
                    'Test': val_metrics[i].get(metric, 0)
                })
    df_evaluation = pd.DataFrame(data_list)
    df_evaluation.to_excel(f'{target}_model_evaluation_data.xlsx', index=False)

def augment_data(X, Y, noise_std=0.01, n_augmentations=2):
    X_aug = [X]
    Y_aug = [Y]
    for _ in range(n_augmentations):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        Y_aug.append(Y)
    return np.concatenate(X_aug), np.concatenate(Y_aug)

def cross_validate(model, X_train, Y_train, X_val, Y_val, n_splits=10, epochs=None, batch_size=None):
    """交叉验证函数，适用于单输出模型（包括传统模型和CNN模型）"""
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    
    # 如果是 DataFrame 则采用 iloc 索引，否则直接索引
    if hasattr(X_train, 'iloc'):
        indexer = lambda X, idx: X.iloc[idx]
    else:
        indexer = lambda X, idx: X[idx]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold+1}/{n_splits}")
        try:
            X_train_fold = indexer(X_train, train_idx)
            Y_train_fold = Y_train[train_idx]
            X_val_fold = indexer(X_train, val_idx)
            Y_val_fold = Y_train[val_idx]
            
            if isinstance(model, nn.Module):
                model_copy = copy.deepcopy(model)
                # 单输出CNN模型的训练
                train_single_model(model_copy, X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, epochs, batch_size)
                train_fold_metrics = evaluate_model(model_copy, X_train_fold, Y_train_fold)
                val_fold_metrics = evaluate_model(model_copy, X_val_fold, Y_val_fold)
            else:
                model_copy = clone(model)
                model_copy.fit(X_train_fold, Y_train_fold)
                train_fold_metrics = evaluate_model(model_copy, X_train_fold, Y_train_fold)
                val_fold_metrics = evaluate_model(model_copy, X_val_fold, Y_val_fold)
            
            for metric, value in train_fold_metrics.items():
                train_metrics[metric].append(value)
            for metric, value in val_fold_metrics.items():
                val_metrics[metric].append(value)
            
            print(f"Fold {fold+1} metrics:")
            print(f"Train: {train_fold_metrics}")
            print(f"Val: {val_fold_metrics}")
        except Exception as e:
            print(f"Error in fold {fold+1}: {str(e)}")
            continue
            
    avg_train = {k: np.mean(v) for k, v in train_metrics.items()}
    avg_val = {k: np.mean(v) for k, v in val_metrics.items()}
    print("\nAverage metrics across all folds:")
    print(f"Train: {avg_train}")
    print(f"Val: {avg_val}")
    return avg_train, avg_val

# 单输出CNN模型
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.1)
        self.num_classes = num_classes

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_single_model(model, X_train, Y_train, X_val, Y_val, epochs=200, batch_size=32, patience=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(device)
    Y_train_tensor = torch.LongTensor(Y_train)
    X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val).to(device)
    Y_val_tensor = torch.LongTensor(Y_val)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            val_loss = criterion(outputs_val, Y_val_tensor).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model)
                break
    return train_losses, val_losses

def plot_all_confusion_matrices(y_true, y_pred_dict, target):
    for model_name, preds in y_pred_dict.items():
        cm = confusion_matrix(y_true, preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name}', fontname='Times New Roman', fontsize=15)
        plt.xlabel('Predicted', fontname='Times New Roman', fontsize=13)
        plt.ylabel('True', fontname='Times New Roman', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{target}_{model_name}_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        # 保存混淆矩阵数据
        df_cm = pd.DataFrame(cm)
        df_cm.to_excel(f'{target}_confusion_matrix_{model_name}.xlsx', index=False)

def plot_roc_curves(y_true, y_pred_proba_dict, target):
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    for (name, y_pred_proba), color in zip(y_pred_proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=color, linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontname='Times New Roman')
    plt.ylabel('True Positive Rate', fontsize=12, fontname='Times New Roman')
    plt.title(f'{target} ROC Curves', fontsize=14, fontname='Times New Roman')
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 10})
    plt.tight_layout()
    plt.savefig(f'{target}_roc_curves.pdf', dpi=300, format='pdf')
    plt.close()
    with pd.ExcelWriter(f'{target}_roc_curves_data.xlsx') as writer:
        for name, y_pred_proba in y_pred_proba_dict.items():
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
            df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
            df.to_excel(writer, sheet_name=name, index=False)

def plot_shap(model, X, feature_names, target, n_samples=300):
    """为单输出分类模型生成与保存SHAP值分析图"""
    print(f"开始为{target}生成SHAP图...")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=42)
    
    try:
        # 使用TreeExplainer创建解释器
        explainer = shap.TreeExplainer(model)
        print(f"SHAP解释器创建成功")
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
        print(f"SHAP shap_values shape: {np.array(shap_values).shape}")
        
        # 如果shap_values是列表（多个类别），则转换为单个数组
        if isinstance(shap_values, list):
            # 转换为形状为(samples, features, classes)的数组
            raw_shap_values = np.stack(shap_values, axis=2)
            print(f"已将列表类型的shap_values转换为数组，shape: {raw_shap_values.shape}")
        else:
            # 如果已经是数组，检查其维度
            raw_shap_values = shap_values
            if len(raw_shap_values.shape) == 2:
                # 如果只有2维(samples, features)，扩展为3维
                raw_shap_values = np.expand_dims(raw_shap_values, axis=2)
                print(f"已将2维shap_values扩展为3维，shape: {raw_shap_values.shape}")
        
        # 确保expected_value是数组类型
        if not isinstance(expected_value, (list, np.ndarray)):
            base_val = np.array([expected_value])
        else:
            base_val = np.array(expected_value)
        
        # 处理每个类别的SHAP值
        num_classes = raw_shap_values.shape[2]
        print(f"处理{num_classes}个类别")
        
        for class_to_show in range(num_classes):
            class_shap_values = raw_shap_values[:, :, class_to_show]
            print(f"处理类别 {class_to_show}, shape: {class_shap_values.shape}")
            
            # 创建Explanation对象
            explanation = shap.Explanation(
                values=class_shap_values,
                base_values=np.full(len(X), base_val[class_to_show]) if len(base_val) > 1 else np.full(len(X), base_val[0]),
                data=X.values,
                feature_names=feature_names
            )
            
            # 绘制beeswarm图并保存
            try:
                plt.figure(figsize=(12, 8))
                shap.plots.beeswarm(explanation, show=False, max_display=10)
                plt.title(f'{target} SHAP Beeswarm - Class {class_to_show}', fontsize=16)
                plt.xlabel('SHAP value (impact on model output)', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{target}_shap_beeswarm_class_{class_to_show}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"已保存beeswarm图: {target}_shap_beeswarm_class_{class_to_show}.pdf")
            except Exception as e:
                print(f"beeswarm图生成失败: {e}")
            
            # 保存beeswarm图的数据到Excel
            try:
                beeswarm_data = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Values': np.mean(np.abs(class_shap_values), axis=0)
                }).sort_values('SHAP_Values', ascending=False)
                beeswarm_data.to_excel(f'{target}_shap_beeswarm_class_{class_to_show}_data.xlsx', index=False)
                print(f"已保存beeswarm数据: {target}_shap_beeswarm_class_{class_to_show}_data.xlsx")
            except Exception as e:
                print(f"保存beeswarm数据失败: {e}")
            
            # 绘制heatmap图并保存
            try:
                plt.figure(figsize=(12, 8))
                shap.plots.heatmap(explanation, show=False, max_display=10)
                plt.title(f'{target} SHAP Heatmap - Class {class_to_show}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{target}_shap_heatmap_class_{class_to_show}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"已保存heatmap图: {target}_shap_heatmap_class_{class_to_show}.pdf")
            except Exception as e:
                print(f"heatmap图生成失败: {e}")
            
            # 保存heatmap图的数据到Excel
            try:
                heatmap_data = pd.DataFrame(
                    class_shap_values,
                    columns=feature_names
                )
                heatmap_data.to_excel(f'{target}_shap_heatmap_class_{class_to_show}_data.xlsx', index=False)
                print(f"已保存heatmap数据: {target}_shap_heatmap_class_{class_to_show}_data.xlsx")
            except Exception as e:
                print(f"保存heatmap数据失败: {e}")
            
            # 绘制bar图并保存
            try:
                plt.figure(figsize=(12, 8))
                shap.plots.bar(explanation, show=False, max_display=10)
                plt.title(f'{target} - SHAP Bar Plot - Class {class_to_show}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{target}_shap_bar_class_{class_to_show}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"已保存bar图: {target}_shap_bar_class_{class_to_show}.pdf")
            except Exception as e:
                print(f"bar图生成失败: {e}")
            
            # 保存bar图的数据到Excel
            try:
                bar_data = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Values': np.mean(np.abs(class_shap_values), axis=0)
                }).sort_values('SHAP_Values', ascending=False)
                bar_data.to_excel(f'{target}_shap_bar_class_{class_to_show}_data.xlsx', index=False)
                print(f"已保存bar数据: {target}_shap_bar_class_{class_to_show}_data.xlsx")
            except Exception as e:
                print(f"保存bar数据失败: {e}")
            
            # 为第一个样本创建Explanation对象用于waterfall图
            sample_explanation = shap.Explanation(
                values=raw_shap_values[0, :, class_to_show],
                base_values=base_val[class_to_show] if len(base_val) > 1 else base_val[0],
                data=X.iloc[0, :].values,
                feature_names=feature_names
            )
            
            # 绘制waterfall图并保存
            try:
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(sample_explanation, show=False)
                plt.title(f'{target} - SHAP Waterfall - Class {class_to_show} (Sample 0)', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{target}_shap_waterfall_class_{class_to_show}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"已保存waterfall图: {target}_shap_waterfall_class_{class_to_show}.pdf")
            except Exception as e:
                print(f"waterfall图生成失败: {e}")
            
            # 保存waterfall图的数据到Excel
            try:
                waterfall_data = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Values': raw_shap_values[0, :, class_to_show]
                }).sort_values('SHAP_Values', key=abs, ascending=False)
                waterfall_data.to_excel(f'{target}_shap_waterfall_class_{class_to_show}_data.xlsx', index=False)
                print(f"已保存waterfall数据: {target}_shap_waterfall_class_{class_to_show}_data.xlsx")
            except Exception as e:
                print(f"保存waterfall数据失败: {e}")
        
        # 特定类别的额外分析（选择默认为class_idx=2或最大可用类索引）
        class_idx = min(2, num_classes-1)
        print(f"对类别{class_idx}进行额外分析")
        
        # 绘制force图并保存
        try:
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                base_val[class_idx] if len(base_val) > 1 else base_val[0],
                raw_shap_values[0, :, class_idx],
                X.iloc[0, :], 
                show=False, 
                matplotlib=True
            )
            plt.title(f'{target} - SHAP Force Plot (Sample 0, class {class_idx})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{target}_shap_force.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存force图: {target}_shap_force.pdf")
        except Exception as e:
            print(f"force图生成失败: {e}")
        
        # 保存force图的数据到Excel
        try:
            force_data = pd.DataFrame({
                'Feature': feature_names,
                'Value': X.iloc[0, :].values,
                'SHAP_Values': raw_shap_values[0, :, class_idx]
            })
            force_data.to_excel(f'{target}_shap_force_data.xlsx', index=False)
            print(f"已保存force数据: {target}_shap_force_data.xlsx")
        except Exception as e:
            print(f"保存force数据失败: {e}")
        
        # 绘制decision图并保存
        try:
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                base_val[class_idx] if len(base_val) > 1 else base_val[0],
                raw_shap_values[:, :, class_idx], 
                X, 
                feature_names=feature_names, 
                show=False
            )
            plt.title(f'{target} - SHAP Decision Plot (class {class_idx})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{target}_shap_decision.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存decision图: {target}_shap_decision.pdf")
        except Exception as e:
            print(f"decision图生成失败: {e}")
        
        # 保存decision图的数据到Excel
        try:
            decision_data = pd.DataFrame(
                raw_shap_values[:, :, class_idx],
                columns=feature_names
            )
            decision_data['sample_index'] = range(len(decision_data))
            decision_data.to_excel(f'{target}_shap_decision_data.xlsx', index=False)
            print(f"已保存decision数据: {target}_shap_decision_data.xlsx")
        except Exception as e:
            print(f"保存decision数据失败: {e}")
        
        # 找到最重要的特征，绘制依赖图并保存
        try:
            feature_importances = np.mean(np.abs(raw_shap_values[:, :, class_idx]), axis=0)
            top_feature_idx = np.argmax(feature_importances)
            top_feature = feature_names[top_feature_idx]
            
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(top_feature, raw_shap_values[:, :, class_idx], X, show=False)
            plt.title(f'{target} - SHAP Dependence Plot (Feature: {top_feature}, class {class_idx})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{target}_shap_dependence.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存dependence图: {target}_shap_dependence.pdf")
        except Exception as e:
            print(f"dependence图生成失败: {e}")
        
        # 保存dependence图的数据到Excel
        try:
            dependence_data = pd.DataFrame({
                'Feature_Value': X[top_feature],
                'SHAP_Value': raw_shap_values[:, top_feature_idx, class_idx]
            })
            dependence_data.to_excel(f'{target}_shap_dependence_data.xlsx', index=False)
            print(f"已保存dependence数据: {target}_shap_dependence_data.xlsx")
        except Exception as e:
            print(f"保存dependence数据失败: {e}")
    
    except Exception as e:
        print(f"SHAP计算出错：{e}")
        import traceback
        traceback.print_exc()
    
    print(f"{target}的SHAP分析完成")

def visualize_cnn_structure(model, input_shape):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, *input_shape).to(device).requires_grad_(True)
    y = model(x)
    model_cpu = model.cpu()
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    with open("cnn_structure.txt", "w") as f:
        f.write(str(model_cpu))
    try:
        dot = make_dot(y_cpu, params=dict(list(model_cpu.named_parameters()) + [('x', x_cpu)]))
        dot.render("cnn_structure", format="png", cleanup=True)
        print("CNN结构图已保存为 cnn_structure.png")
    except Exception as e:
        print("无法生成CNN结构图，可能是因为未安装Graphviz。")
        print("模型结构已保存到 cnn_structure.txt")
        print(f"错误信息: {str(e)}")

# 添加一个新函数，用于绘制所有人群和模型的混淆矩阵
def plot_all_population_model_confusion_matrices(prediction_data, model_names, population_names):
    """
    绘制4行6列的混淆矩阵图，行是4个人群类别，列是6个模型。
    
    参数:
    prediction_data: 字典，格式为 {population_name: {model_name: {'y_true': array, 'y_pred': array}}}
    model_names: 模型名称列表，按顺序排列
    population_names: 人群名称列表，按顺序排列
    """
    # 创建一个大图
    fig = plt.figure(figsize=(26, 20))
    
    # 使用GridSpec来控制子图布局
    gs = gridspec.GridSpec(4, 6, figure=fig, wspace=0.3, hspace=0.3)
    
    # 保存所有混淆矩阵数据，用于导出到Excel
    all_cm_data = []
    
    # 遍历所有人群和模型
    for i, population in enumerate(population_names):
        for j, model in enumerate(model_names):
            # 获取该人群和模型的预测数据
            y_true = prediction_data[population][model]['y_true']
            y_pred = prediction_data[population][model]['y_pred']
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            
            # 添加到数据列表
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    all_cm_data.append({
                        'Population': population,
                        'Model': model,
                        'True_Label': ii,
                        'Predicted_Label': jj,
                        'Count': cm[ii, jj]
                    })
            
            # 创建子图
            ax = fig.add_subplot(gs[i, j])
            
            # 绘制混淆矩阵热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        cbar=False, square=True)
            
            # 设置标题和标签
            if i == 0:  # 第一行设置模型名称
                ax.set_title(model, fontsize=14, fontname='Times New Roman')
            
            if j == 0:  # 第一列设置人群名称
                ax.set_ylabel(population.replace('_', ' '), fontsize=14, fontname='Times New Roman')
                
            # 仅在最底部的图上显示x标签
            if i == len(population_names) - 1:
                ax.set_xlabel('Predicted', fontsize=12, fontname='Times New Roman')
            else:
                ax.set_xlabel('')
                
            # 如果不是第一列，不显示y轴标签
            if j != 0:
                ax.set_ylabel('')
                
            # 设置刻度标签
            ax.set_xticklabels(range(4), fontsize=10, fontname='Times New Roman')
            ax.set_yticklabels(range(4), fontsize=10, fontname='Times New Roman', rotation=0)
    
    # 设置整体标题
    plt.suptitle('Confusion Matrices for All Populations and Models', 
                 fontsize=20, fontname='Times New Roman', y=0.98)
    
    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('all_population_model_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('all_population_model_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据到Excel
    df_cm = pd.DataFrame(all_cm_data)
    df_cm.to_excel('all_confusion_matrices_data.xlsx', index=False)
    
    # 也保存为透视表格式，更方便查看
    pivot_cm = df_cm.pivot_table(
        index=['Population', 'True_Label'],
        columns=['Model', 'Predicted_Label'],
        values='Count',
        aggfunc='sum'
    ).fillna(0).astype(int)
    
    pivot_cm.to_excel('all_confusion_matrices_pivot.xlsx')
    
    print("所有人群和模型的混淆矩阵已保存。")

##########################################
# 主程序：分别对城市男性、城市女性、农村男性、农村女性进行建模、评估和SHAP分析
##########################################

# 创建一个字典来存储所有模型在测试集上的预测结果
all_predictions = {
    'Urban_Male': {},
    'Urban_Female': {},
    'Rural_Male': {},
    'Rural_Female': {}
}

###########################
# 城市男性模型部分
###########################
target_urban_male = "Urban_Male_Model"

print("\n=== 使用贝叶斯优化训练城市男性模型 ===")
# 使用贝叶斯优化来优化RandomForest模型
print("优化 RandomForest 模型...")
urban_male_rf = optimize_rf(X_tree_urban_male_train, Y_urban_male_train, n_iter=10)

# SVM模型
urban_male_svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=42)

# 使用贝叶斯优化来优化XGBoost模型
print("优化 XGBoost 模型...")
urban_male_xgb = optimize_xgb(X_tree_urban_male_train, Y_urban_male_train, n_iter=10)

# 使用贝叶斯优化来优化GBDT模型
print("优化 GBDT 模型...")
urban_male_gbdt = optimize_gbdt(X_tree_urban_male_train_imputed, Y_urban_male_train, n_iter=10)

# 使用贝叶斯优化来优化LightGBM模型
print("优化 LightGBM 模型...")
urban_male_lgbm = optimize_lgbm(X_tree_urban_male_train, Y_urban_male_train, n_iter=10)

print("\n=== 城市男性模型优化完成，开始评估 ===")
urban_male_rf_metrics = evaluate_model(urban_male_rf, X_tree_urban_male_val, Y_urban_male_val)
urban_male_svm_metrics = evaluate_model(urban_male_svm.fit(X_non_tree_urban_male_train, Y_urban_male_train), X_non_tree_urban_male_val, Y_urban_male_val)
urban_male_xgb_metrics = evaluate_model(urban_male_xgb, X_tree_urban_male_val, Y_urban_male_val)
urban_male_gbdt_metrics = evaluate_model(urban_male_gbdt, X_tree_urban_male_val_imputed, Y_urban_male_val)
urban_male_lgbm_metrics = evaluate_model(urban_male_lgbm, X_tree_urban_male_val, Y_urban_male_val)

print("Urban Male RF Metrics:", urban_male_rf_metrics)
print("Urban Male SVM Metrics:", urban_male_svm_metrics)
print("Urban Male XGBoost Metrics:", urban_male_xgb_metrics)
print("Urban Male GBDT Metrics:", urban_male_gbdt_metrics)
print("Urban Male LightGBM Metrics:", urban_male_lgbm_metrics)

print("\n=== Cross-validation for 城市男性模型 ===")
urban_male_rf_cv_train, urban_male_rf_cv_val = cross_validate(urban_male_rf, X_tree_urban_male_train, Y_urban_male_train, X_tree_urban_male_val, Y_urban_male_val, n_splits=10)
urban_male_svm_cv_train, urban_male_svm_cv_val = cross_validate(urban_male_svm, X_non_tree_urban_male_train, Y_urban_male_train, X_non_tree_urban_male_val, Y_urban_male_val, n_splits=10)
urban_male_xgb_cv_train, urban_male_xgb_cv_val = cross_validate(urban_male_xgb, X_tree_urban_male_train, Y_urban_male_train, X_tree_urban_male_val, Y_urban_male_val, n_splits=10)
urban_male_gbdt_cv_train, urban_male_gbdt_cv_val = cross_validate(urban_male_gbdt, X_tree_urban_male_train_imputed, Y_urban_male_train, X_tree_urban_male_val_imputed, Y_urban_male_val, n_splits=10)
urban_male_lgbm_cv_train, urban_male_lgbm_cv_val = cross_validate(urban_male_lgbm, X_tree_urban_male_train, Y_urban_male_train, X_tree_urban_male_val, Y_urban_male_val, n_splits=10)

# 构建城市男性CNN模型
urban_male_cnn = CNN(input_dim=X_non_tree_urban_male_train.shape[1], num_classes=4)
print("\n=== Cross-validation for 城市男性CNN模型 ===")
urban_male_cnn_cv_train, urban_male_cnn_cv_val = cross_validate(urban_male_cnn, X_non_tree_urban_male_train, Y_urban_male_train, X_non_tree_urban_male_val, Y_urban_male_val, n_splits=10, epochs=200, batch_size=32)
print("城市男性CNN Cross-validation Training Metrics:", urban_male_cnn_cv_train)
print("城市男性CNN Cross-validation Validation Metrics:", urban_male_cnn_cv_val)

urban_male_cnn_train_losses, urban_male_cnn_val_losses = train_single_model(urban_male_cnn, X_non_tree_urban_male_train, Y_urban_male_train, 
                                                               X_non_tree_urban_male_val, Y_urban_male_val, epochs=200, batch_size=32, patience=10)
save_cnn_model(urban_male_cnn, 'urban_male_cnn_model.pth')
urban_male_cnn_metrics = evaluate_model(urban_male_cnn, X_non_tree_urban_male_test, Y_urban_male_test)
print("城市男性CNN Test Metrics:", urban_male_cnn_metrics)

# 绘制城市男性CNN收敛曲线
plot_cnn_convergence(urban_male_cnn_train_losses, urban_male_cnn_val_losses, target_urban_male)

# 绘制城市男性模型评估图
# 评估城市男性模型在训练集上的性能
urban_male_cnn.eval()
urban_male_rf.fit(X_tree_urban_male_train, Y_urban_male_train)
urban_male_svm.fit(X_non_tree_urban_male_train, Y_urban_male_train) 
urban_male_xgb.fit(X_tree_urban_male_train, Y_urban_male_train)
urban_male_gbdt.fit(X_tree_urban_male_train_imputed, Y_urban_male_train)
urban_male_lgbm.fit(X_tree_urban_male_train, Y_urban_male_train)

urban_male_cnn_train_metrics = evaluate_model(urban_male_cnn, X_non_tree_urban_male_train, Y_urban_male_train)
urban_male_rf_train_metrics = evaluate_model(urban_male_rf, X_tree_urban_male_train, Y_urban_male_train)
urban_male_svm_train_metrics = evaluate_model(urban_male_svm, X_non_tree_urban_male_train, Y_urban_male_train)
urban_male_xgb_train_metrics = evaluate_model(urban_male_xgb, X_tree_urban_male_train, Y_urban_male_train)
urban_male_gbdt_train_metrics = evaluate_model(urban_male_gbdt, X_tree_urban_male_train_imputed, Y_urban_male_train)
urban_male_lgbm_train_metrics = evaluate_model(urban_male_lgbm, X_tree_urban_male_train, Y_urban_male_train)

# 评估城市男性模型在测试集上的性能
urban_male_cnn_test_metrics = evaluate_model(urban_male_cnn, X_non_tree_urban_male_test, Y_urban_male_test)
urban_male_rf_test_metrics = evaluate_model(urban_male_rf, X_tree_urban_male_test, Y_urban_male_test)
urban_male_svm_test_metrics = evaluate_model(urban_male_svm, X_non_tree_urban_male_test, Y_urban_male_test)
urban_male_xgb_test_metrics = evaluate_model(urban_male_xgb, X_tree_urban_male_test, Y_urban_male_test)
urban_male_gbdt_test_metrics = evaluate_model(urban_male_gbdt, X_tree_urban_male_test_imputed, Y_urban_male_test)
urban_male_lgbm_test_metrics = evaluate_model(urban_male_lgbm, X_tree_urban_male_test, Y_urban_male_test)

# 创建训练集和测试集性能指标列表
urban_male_model_train_metrics = [urban_male_cnn_train_metrics, urban_male_rf_train_metrics, urban_male_svm_train_metrics, urban_male_xgb_train_metrics, urban_male_gbdt_train_metrics, urban_male_lgbm_train_metrics]
urban_male_model_test_metrics = [urban_male_cnn_test_metrics, urban_male_rf_test_metrics, urban_male_svm_test_metrics, urban_male_xgb_test_metrics, urban_male_gbdt_test_metrics, urban_male_lgbm_test_metrics]

# 绘制城市男性模型评估图 - 使用区分的训练集和测试集指标
plot_model_evaluation(urban_male_model_train_metrics, urban_male_model_test_metrics, target_urban_male)


# 生成模型在测试集上的预测
# CNN预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_urban_male_tensor = torch.FloatTensor(X_non_tree_urban_male_test.values if hasattr(X_non_tree_urban_male_test, 'values') else X_non_tree_urban_male_test).to(device)
urban_male_cnn.eval()
with torch.no_grad():
    outputs = urban_male_cnn(X_urban_male_tensor)
    urban_male_cnn_preds = torch.max(outputs, 1)[1].cpu().numpy()
    urban_male_cnn_probas = torch.softmax(outputs, dim=1).cpu().numpy()

# 其他模型的预测
urban_male_rf.fit(X_tree_urban_male_train, Y_urban_male_train)
urban_male_rf_preds = urban_male_rf.predict(X_tree_urban_male_test)
urban_male_rf_probas = urban_male_rf.predict_proba(X_tree_urban_male_test)

urban_male_svm.fit(X_non_tree_urban_male_train, Y_urban_male_train)
urban_male_svm_preds = urban_male_svm.predict(X_non_tree_urban_male_test)
urban_male_svm_probas = urban_male_svm.predict_proba(X_non_tree_urban_male_test)

urban_male_xgb.fit(X_tree_urban_male_train, Y_urban_male_train)
urban_male_xgb_preds = urban_male_xgb.predict(X_tree_urban_male_test)
urban_male_xgb_probas = urban_male_xgb.predict_proba(X_tree_urban_male_test)

urban_male_gbdt.fit(X_tree_urban_male_train_imputed, Y_urban_male_train)
urban_male_gbdt_preds = urban_male_gbdt.predict(X_tree_urban_male_test_imputed)
urban_male_gbdt_probas = urban_male_gbdt.predict_proba(X_tree_urban_male_test_imputed)

urban_male_lgbm.fit(X_tree_urban_male_train, Y_urban_male_train)
urban_male_lgbm_preds = urban_male_lgbm.predict(X_tree_urban_male_test)
urban_male_lgbm_probas = urban_male_lgbm.predict_proba(X_tree_urban_male_test)

# 存储预测结果
all_predictions['Urban_Male']['CNN'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_cnn_preds}
all_predictions['Urban_Male']['RF'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_rf_preds}
all_predictions['Urban_Male']['SVM'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_svm_preds}
all_predictions['Urban_Male']['XGBoost'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_xgb_preds}
all_predictions['Urban_Male']['GBDT'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_gbdt_preds}
all_predictions['Urban_Male']['LightGBM'] = {'y_true': Y_urban_male_test, 'y_pred': urban_male_lgbm_preds}

# 绘制城市男性混淆矩阵（单个模型）
plot_all_confusion_matrices(Y_urban_male_test, {"Urban_Male_LightGBM": urban_male_lgbm_preds}, target_urban_male)

# 绘制城市男性各模型的ROC曲线
roc_dict_urban_male = {
    "Urban Male RF": urban_male_rf_probas,
    "Urban Male XGBoost": urban_male_xgb_probas,
    "Urban Male GBDT": urban_male_gbdt_probas,
    "Urban Male LightGBM": urban_male_lgbm_probas,
    "Urban Male SVM": urban_male_svm_probas,
    "Urban Male CNN": urban_male_cnn_probas
}
plot_roc_curves(Y_urban_male_test, roc_dict_urban_male, target_urban_male)

# SHAP 分析：以urban_male_lgbm为例
urban_male_lgbm.fit(X_tree_urban_male_train, Y_urban_male_train)
X_train_urban_male_shap = X_tree_urban_male_train.copy()
plot_shap(urban_male_lgbm, X_train_urban_male_shap, X_train_urban_male_shap.columns.tolist(), target_urban_male, n_samples=500)

# CNN结构可视化
visualize_cnn_structure(urban_male_cnn, (X_non_tree_urban_male_train.shape[1],))

###########################
# 城市女性模型部分
###########################
target_urban_female = "Urban_Female_Model"

print("\n=== 使用贝叶斯优化训练城市女性模型 ===")
# 使用贝叶斯优化来优化RandomForest模型
print("优化 RandomForest 模型...")
urban_female_rf = optimize_rf(X_tree_urban_female_train, Y_urban_female_train, n_iter=10)

# SVM模型
urban_female_svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=42)

# 使用贝叶斯优化来优化XGBoost模型
print("优化 XGBoost 模型...")
urban_female_xgb = optimize_xgb(X_tree_urban_female_train, Y_urban_female_train, n_iter=10)

# 使用贝叶斯优化来优化GBDT模型
print("优化 GBDT 模型...")
urban_female_gbdt = optimize_gbdt(X_tree_urban_female_train_imputed, Y_urban_female_train, n_iter=10)

# 使用贝叶斯优化来优化LightGBM模型
print("优化 LightGBM 模型...")
urban_female_lgbm = optimize_lgbm(X_tree_urban_female_train, Y_urban_female_train, n_iter=10)

print("\n=== 城市女性模型优化完成，开始评估 ===")
urban_female_rf_metrics = evaluate_model(urban_female_rf, X_tree_urban_female_val, Y_urban_female_val)
urban_female_svm_metrics = evaluate_model(urban_female_svm.fit(X_non_tree_urban_female_train, Y_urban_female_train), X_non_tree_urban_female_val, Y_urban_female_val)
urban_female_xgb_metrics = evaluate_model(urban_female_xgb, X_tree_urban_female_val, Y_urban_female_val)
urban_female_gbdt_metrics = evaluate_model(urban_female_gbdt, X_tree_urban_female_val_imputed, Y_urban_female_val)
urban_female_lgbm_metrics = evaluate_model(urban_female_lgbm, X_tree_urban_female_val, Y_urban_female_val)

print("Urban Female RF Metrics:", urban_female_rf_metrics)
print("Urban Female SVM Metrics:", urban_female_svm_metrics)
print("Urban Female XGBoost Metrics:", urban_female_xgb_metrics)
print("Urban Female GBDT Metrics:", urban_female_gbdt_metrics)
print("Urban Female LightGBM Metrics:", urban_female_lgbm_metrics)

print("\n=== Cross-validation for 城市女性模型 ===")
urban_female_rf_cv_train, urban_female_rf_cv_val = cross_validate(urban_female_rf, X_tree_urban_female_train, Y_urban_female_train, X_tree_urban_female_val, Y_urban_female_val, n_splits=10)
urban_female_svm_cv_train, urban_female_svm_cv_val = cross_validate(urban_female_svm, X_non_tree_urban_female_train, Y_urban_female_train, X_non_tree_urban_female_val, Y_urban_female_val, n_splits=10)
urban_female_xgb_cv_train, urban_female_xgb_cv_val = cross_validate(urban_female_xgb, X_tree_urban_female_train, Y_urban_female_train, X_tree_urban_female_val, Y_urban_female_val, n_splits=10)
urban_female_gbdt_cv_train, urban_female_gbdt_cv_val = cross_validate(urban_female_gbdt, X_tree_urban_female_train_imputed, Y_urban_female_train, X_tree_urban_female_val_imputed, Y_urban_female_val, n_splits=10)
urban_female_lgbm_cv_train, urban_female_lgbm_cv_val = cross_validate(urban_female_lgbm, X_tree_urban_female_train, Y_urban_female_train, X_tree_urban_female_val, Y_urban_female_val, n_splits=10)

# 构建城市女性CNN模型
urban_female_cnn = CNN(input_dim=X_non_tree_urban_female_train.shape[1], num_classes=4)
print("\n=== Cross-validation for 城市女性CNN模型 ===")
urban_female_cnn_cv_train, urban_female_cnn_cv_val = cross_validate(urban_female_cnn, X_non_tree_urban_female_train, Y_urban_female_train, X_non_tree_urban_female_val, Y_urban_female_val, n_splits=10, epochs=200, batch_size=32)
print("城市女性CNN Cross-validation Training Metrics:", urban_female_cnn_cv_train)
print("城市女性CNN Cross-validation Validation Metrics:", urban_female_cnn_cv_val)

urban_female_cnn_train_losses, urban_female_cnn_val_losses = train_single_model(urban_female_cnn, X_non_tree_urban_female_train, Y_urban_female_train, 
                                                               X_non_tree_urban_female_val, Y_urban_female_val, epochs=200, batch_size=32, patience=10)
save_cnn_model(urban_female_cnn, 'urban_female_cnn_model.pth')
urban_female_cnn_metrics = evaluate_model(urban_female_cnn, X_non_tree_urban_female_test, Y_urban_female_test)
print("城市女性CNN Test Metrics:", urban_female_cnn_metrics)

# 绘制城市女性CNN收敛曲线
plot_cnn_convergence(urban_female_cnn_train_losses, urban_female_cnn_val_losses, target_urban_female)

# 绘制城市女性模型评估图
# 评估城市女性模型在训练集上的性能
urban_female_cnn.eval()
urban_female_rf.fit(X_tree_urban_female_train, Y_urban_female_train)
urban_female_svm.fit(X_non_tree_urban_female_train, Y_urban_female_train) 
urban_female_xgb.fit(X_tree_urban_female_train, Y_urban_female_train)
urban_female_gbdt.fit(X_tree_urban_female_train_imputed, Y_urban_female_train)
urban_female_lgbm.fit(X_tree_urban_female_train, Y_urban_female_train)

urban_female_cnn_train_metrics = evaluate_model(urban_female_cnn, X_non_tree_urban_female_train, Y_urban_female_train)
urban_female_rf_train_metrics = evaluate_model(urban_female_rf, X_tree_urban_female_train, Y_urban_female_train)
urban_female_svm_train_metrics = evaluate_model(urban_female_svm, X_non_tree_urban_female_train, Y_urban_female_train)
urban_female_xgb_train_metrics = evaluate_model(urban_female_xgb, X_tree_urban_female_train, Y_urban_female_train)
urban_female_gbdt_train_metrics = evaluate_model(urban_female_gbdt, X_tree_urban_female_train_imputed, Y_urban_female_train)
urban_female_lgbm_train_metrics = evaluate_model(urban_female_lgbm, X_tree_urban_female_train, Y_urban_female_train)

# 评估城市女性模型在测试集上的性能
urban_female_cnn_test_metrics = evaluate_model(urban_female_cnn, X_non_tree_urban_female_test, Y_urban_female_test)
urban_female_rf_test_metrics = evaluate_model(urban_female_rf, X_tree_urban_female_test, Y_urban_female_test)
urban_female_svm_test_metrics = evaluate_model(urban_female_svm, X_non_tree_urban_female_test, Y_urban_female_test)
urban_female_xgb_test_metrics = evaluate_model(urban_female_xgb, X_tree_urban_female_test, Y_urban_female_test)
urban_female_gbdt_test_metrics = evaluate_model(urban_female_gbdt, X_tree_urban_female_test_imputed, Y_urban_female_test)
urban_female_lgbm_test_metrics = evaluate_model(urban_female_lgbm, X_tree_urban_female_test, Y_urban_female_test)

# 创建训练集和测试集性能指标列表
urban_female_model_train_metrics = [urban_female_cnn_train_metrics, urban_female_rf_train_metrics, urban_female_svm_train_metrics, urban_female_xgb_train_metrics, urban_female_gbdt_train_metrics, urban_female_lgbm_train_metrics]
urban_female_model_test_metrics = [urban_female_cnn_test_metrics, urban_female_rf_test_metrics, urban_female_svm_test_metrics, urban_female_xgb_test_metrics, urban_female_gbdt_test_metrics, urban_female_lgbm_test_metrics]

# 绘制城市女性模型评估图 - 使用区分的训练集和测试集指标
plot_model_evaluation(urban_female_model_train_metrics, urban_female_model_test_metrics, target_urban_female)


# 生成模型在测试集上的预测
# CNN预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_urban_female_tensor = torch.FloatTensor(X_non_tree_urban_female_test.values if hasattr(X_non_tree_urban_female_test, 'values') else X_non_tree_urban_female_test).to(device)
urban_female_cnn.eval()
with torch.no_grad():
    outputs = urban_female_cnn(X_urban_female_tensor)
    urban_female_cnn_preds = torch.max(outputs, 1)[1].cpu().numpy()
    urban_female_cnn_probas = torch.softmax(outputs, dim=1).cpu().numpy()

# 其他模型的预测
urban_female_rf.fit(X_tree_urban_female_train, Y_urban_female_train)
urban_female_rf_preds = urban_female_rf.predict(X_tree_urban_female_test)
urban_female_rf_probas = urban_female_rf.predict_proba(X_tree_urban_female_test)

urban_female_svm.fit(X_non_tree_urban_female_train, Y_urban_female_train)
urban_female_svm_preds = urban_female_svm.predict(X_non_tree_urban_female_test)
urban_female_svm_probas = urban_female_svm.predict_proba(X_non_tree_urban_female_test)

urban_female_xgb.fit(X_tree_urban_female_train, Y_urban_female_train)
urban_female_xgb_preds = urban_female_xgb.predict(X_tree_urban_female_test)
urban_female_xgb_probas = urban_female_xgb.predict_proba(X_tree_urban_female_test)

urban_female_gbdt.fit(X_tree_urban_female_train_imputed, Y_urban_female_train)
urban_female_gbdt_preds = urban_female_gbdt.predict(X_tree_urban_female_test_imputed)
urban_female_gbdt_probas = urban_female_gbdt.predict_proba(X_tree_urban_female_test_imputed)

urban_female_lgbm.fit(X_tree_urban_female_train, Y_urban_female_train)
urban_female_lgbm_preds = urban_female_lgbm.predict(X_tree_urban_female_test)
urban_female_lgbm_probas = urban_female_lgbm.predict_proba(X_tree_urban_female_test)

# 存储预测结果
all_predictions['Urban_Female']['CNN'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_cnn_preds}
all_predictions['Urban_Female']['RF'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_rf_preds}
all_predictions['Urban_Female']['SVM'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_svm_preds}
all_predictions['Urban_Female']['XGBoost'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_xgb_preds}
all_predictions['Urban_Female']['GBDT'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_gbdt_preds}
all_predictions['Urban_Female']['LightGBM'] = {'y_true': Y_urban_female_test, 'y_pred': urban_female_lgbm_preds}

# 绘制城市女性模型混淆矩阵（单个模型）
plot_all_confusion_matrices(Y_urban_female_test, {"Urban_Female_LightGBM": urban_female_lgbm_preds}, target_urban_female)

# 绘制城市女性各模型的ROC曲线
roc_dict_urban_female = {
    "Urban Female RF": urban_female_rf_probas,
    "Urban Female XGBoost": urban_female_xgb_probas,
    "Urban Female GBDT": urban_female_gbdt_probas,
    "Urban Female LightGBM": urban_female_lgbm_probas,
    "Urban Female SVM": urban_female_svm_probas,
    "Urban Female CNN": urban_female_cnn_probas
}
plot_roc_curves(Y_urban_female_test, roc_dict_urban_female, target_urban_female)

# SHAP 分析：以urban_female_lgbm为例
urban_female_lgbm.fit(X_tree_urban_female_train, Y_urban_female_train)
X_train_urban_female_shap = X_tree_urban_female_train.copy()
plot_shap(urban_female_lgbm, X_train_urban_female_shap, X_train_urban_female_shap.columns.tolist(), target_urban_female, n_samples=500)

###########################
# 农村男性模型部分
###########################
target_rural_male = "Rural_Male_Model"

print("\n=== 使用贝叶斯优化训练农村男性模型 ===")
# 使用贝叶斯优化来优化RandomForest模型
print("优化 RandomForest 模型...")
rural_male_rf = optimize_rf(X_tree_rural_male_train, Y_rural_male_train, n_iter=10)

# SVM模型
rural_male_svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=42)

# 使用贝叶斯优化来优化XGBoost模型
print("优化 XGBoost 模型...")
rural_male_xgb = optimize_xgb(X_tree_rural_male_train, Y_rural_male_train, n_iter=10)

# 使用贝叶斯优化来优化GBDT模型
print("优化 GBDT 模型...")
rural_male_gbdt = optimize_gbdt(X_tree_rural_male_train_imputed, Y_rural_male_train, n_iter=10)

# 使用贝叶斯优化来优化LightGBM模型
print("优化 LightGBM 模型...")
rural_male_lgbm = optimize_lgbm(X_tree_rural_male_train, Y_rural_male_train, n_iter=10)

print("\n=== 农村男性模型优化完成，开始评估 ===")
rural_male_rf_metrics = evaluate_model(rural_male_rf, X_tree_rural_male_val, Y_rural_male_val)
rural_male_svm_metrics = evaluate_model(rural_male_svm.fit(X_non_tree_rural_male_train, Y_rural_male_train), X_non_tree_rural_male_val, Y_rural_male_val)
rural_male_xgb_metrics = evaluate_model(rural_male_xgb, X_tree_rural_male_val, Y_rural_male_val)
rural_male_gbdt_metrics = evaluate_model(rural_male_gbdt, X_tree_rural_male_val_imputed, Y_rural_male_val)
rural_male_lgbm_metrics = evaluate_model(rural_male_lgbm, X_tree_rural_male_val, Y_rural_male_val)

print("Rural Male RF Metrics:", rural_male_rf_metrics)
print("Rural Male SVM Metrics:", rural_male_svm_metrics)
print("Rural Male XGBoost Metrics:", rural_male_xgb_metrics)
print("Rural Male GBDT Metrics:", rural_male_gbdt_metrics)
print("Rural Male LightGBM Metrics:", rural_male_lgbm_metrics)

print("\n=== Cross-validation for 农村男性模型 ===")
rural_male_rf_cv_train, rural_male_rf_cv_val = cross_validate(rural_male_rf, X_tree_rural_male_train, Y_rural_male_train, X_tree_rural_male_val, Y_rural_male_val, n_splits=10)
rural_male_svm_cv_train, rural_male_svm_cv_val = cross_validate(rural_male_svm, X_non_tree_rural_male_train, Y_rural_male_train, X_non_tree_rural_male_val, Y_rural_male_val, n_splits=10)
rural_male_xgb_cv_train, rural_male_xgb_cv_val = cross_validate(rural_male_xgb, X_tree_rural_male_train, Y_rural_male_train, X_tree_rural_male_val, Y_rural_male_val, n_splits=10)
rural_male_gbdt_cv_train, rural_male_gbdt_cv_val = cross_validate(rural_male_gbdt, X_tree_rural_male_train_imputed, Y_rural_male_train, X_tree_rural_male_val_imputed, Y_rural_male_val, n_splits=10)
rural_male_lgbm_cv_train, rural_male_lgbm_cv_val = cross_validate(rural_male_lgbm, X_tree_rural_male_train, Y_rural_male_train, X_tree_rural_male_val, Y_rural_male_val, n_splits=10)

# 构建农村男性CNN模型
rural_male_cnn = CNN(input_dim=X_non_tree_rural_male_train.shape[1], num_classes=4)
print("\n=== Cross-validation for 农村男性CNN模型 ===")
rural_male_cnn_cv_train, rural_male_cnn_cv_val = cross_validate(rural_male_cnn, X_non_tree_rural_male_train, Y_rural_male_train, X_non_tree_rural_male_val, Y_rural_male_val, n_splits=10, epochs=200, batch_size=32)
print("农村男性CNN Cross-validation Training Metrics:", rural_male_cnn_cv_train)
print("农村男性CNN Cross-validation Validation Metrics:", rural_male_cnn_cv_val)

rural_male_cnn_train_losses, rural_male_cnn_val_losses = train_single_model(rural_male_cnn, X_non_tree_rural_male_train, Y_rural_male_train, 
                                                               X_non_tree_rural_male_val, Y_rural_male_val, epochs=200, batch_size=32, patience=10)
save_cnn_model(rural_male_cnn, 'rural_male_cnn_model.pth')
rural_male_cnn_metrics = evaluate_model(rural_male_cnn, X_non_tree_rural_male_test, Y_rural_male_test)
print("农村男性CNN Test Metrics:", rural_male_cnn_metrics)

# 绘制农村男性CNN收敛曲线
plot_cnn_convergence(rural_male_cnn_train_losses, rural_male_cnn_val_losses, target_rural_male)

# 绘制农村男性模型评估图
# 评估农村男性模型在训练集上的性能
rural_male_cnn.eval()
rural_male_rf.fit(X_tree_rural_male_train, Y_rural_male_train)
rural_male_svm.fit(X_non_tree_rural_male_train, Y_rural_male_train) 
rural_male_xgb.fit(X_tree_rural_male_train, Y_rural_male_train)
rural_male_gbdt.fit(X_tree_rural_male_train_imputed, Y_rural_male_train)
rural_male_lgbm.fit(X_tree_rural_male_train, Y_rural_male_train)

rural_male_cnn_train_metrics = evaluate_model(rural_male_cnn, X_non_tree_rural_male_train, Y_rural_male_train)
rural_male_rf_train_metrics = evaluate_model(rural_male_rf, X_tree_rural_male_train, Y_rural_male_train)
rural_male_svm_train_metrics = evaluate_model(rural_male_svm, X_non_tree_rural_male_train, Y_rural_male_train)
rural_male_xgb_train_metrics = evaluate_model(rural_male_xgb, X_tree_rural_male_train, Y_rural_male_train)
rural_male_gbdt_train_metrics = evaluate_model(rural_male_gbdt, X_tree_rural_male_train_imputed, Y_rural_male_train)
rural_male_lgbm_train_metrics = evaluate_model(rural_male_lgbm, X_tree_rural_male_train, Y_rural_male_train)

# 评估农村男性模型在测试集上的性能
rural_male_cnn_test_metrics = evaluate_model(rural_male_cnn, X_non_tree_rural_male_test, Y_rural_male_test)
rural_male_rf_test_metrics = evaluate_model(rural_male_rf, X_tree_rural_male_test, Y_rural_male_test)
rural_male_svm_test_metrics = evaluate_model(rural_male_svm, X_non_tree_rural_male_test, Y_rural_male_test)
rural_male_xgb_test_metrics = evaluate_model(rural_male_xgb, X_tree_rural_male_test, Y_rural_male_test)
rural_male_gbdt_test_metrics = evaluate_model(rural_male_gbdt, X_tree_rural_male_test_imputed, Y_rural_male_test)
rural_male_lgbm_test_metrics = evaluate_model(rural_male_lgbm, X_tree_rural_male_test, Y_rural_male_test)

# 创建训练集和测试集性能指标列表
rural_male_model_train_metrics = [rural_male_cnn_train_metrics, rural_male_rf_train_metrics, rural_male_svm_train_metrics, rural_male_xgb_train_metrics, rural_male_gbdt_train_metrics, rural_male_lgbm_train_metrics]
rural_male_model_test_metrics = [rural_male_cnn_test_metrics, rural_male_rf_test_metrics, rural_male_svm_test_metrics, rural_male_xgb_test_metrics, rural_male_gbdt_test_metrics, rural_male_lgbm_test_metrics]

# 绘制农村男性模型评估图 - 使用区分的训练集和测试集指标
plot_model_evaluation(rural_male_model_train_metrics, rural_male_model_test_metrics, target_rural_male)

# 生成模型在测试集上的预测
# CNN预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_rural_male_tensor = torch.FloatTensor(X_non_tree_rural_male_test.values if hasattr(X_non_tree_rural_male_test, 'values') else X_non_tree_rural_male_test).to(device)
rural_male_cnn.eval()
with torch.no_grad():
    outputs = rural_male_cnn(X_rural_male_tensor)
    rural_male_cnn_preds = torch.max(outputs, 1)[1].cpu().numpy()
    rural_male_cnn_probas = torch.softmax(outputs, dim=1).cpu().numpy()

# 其他模型的预测
rural_male_rf.fit(X_tree_rural_male_train, Y_rural_male_train)
rural_male_rf_preds = rural_male_rf.predict(X_tree_rural_male_test)
rural_male_rf_probas = rural_male_rf.predict_proba(X_tree_rural_male_test)

rural_male_svm.fit(X_non_tree_rural_male_train, Y_rural_male_train)
rural_male_svm_preds = rural_male_svm.predict(X_non_tree_rural_male_test)
rural_male_svm_probas = rural_male_svm.predict_proba(X_non_tree_rural_male_test)

rural_male_xgb.fit(X_tree_rural_male_train, Y_rural_male_train)
rural_male_xgb_preds = rural_male_xgb.predict(X_tree_rural_male_test)
rural_male_xgb_probas = rural_male_xgb.predict_proba(X_tree_rural_male_test)

rural_male_gbdt.fit(X_tree_rural_male_train_imputed, Y_rural_male_train)
rural_male_gbdt_preds = rural_male_gbdt.predict(X_tree_rural_male_test_imputed)
rural_male_gbdt_probas = rural_male_gbdt.predict_proba(X_tree_rural_male_test_imputed)

rural_male_lgbm.fit(X_tree_rural_male_train, Y_rural_male_train)
rural_male_lgbm_preds = rural_male_lgbm.predict(X_tree_rural_male_test)
rural_male_lgbm_probas = rural_male_lgbm.predict_proba(X_tree_rural_male_test)

# 存储预测结果
all_predictions['Rural_Male']['CNN'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_cnn_preds}
all_predictions['Rural_Male']['RF'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_rf_preds}
all_predictions['Rural_Male']['SVM'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_svm_preds}
all_predictions['Rural_Male']['XGBoost'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_xgb_preds}
all_predictions['Rural_Male']['GBDT'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_gbdt_preds}
all_predictions['Rural_Male']['LightGBM'] = {'y_true': Y_rural_male_test, 'y_pred': rural_male_lgbm_preds}

# 绘制农村男性模型混淆矩阵（单个模型）
plot_all_confusion_matrices(Y_rural_male_test, {"Rural_Male_LightGBM": rural_male_lgbm_preds}, target_rural_male)

# 绘制农村男性各模型的ROC曲线
roc_dict_rural_male = {
    "Rural Male RF": rural_male_rf_probas,
    "Rural Male XGBoost": rural_male_xgb_probas,
    "Rural Male GBDT": rural_male_gbdt_probas,
    "Rural Male LightGBM": rural_male_lgbm_probas,
    "Rural Male SVM": rural_male_svm_probas,
    "Rural Male CNN": rural_male_cnn_probas
}
plot_roc_curves(Y_rural_male_test, roc_dict_rural_male, target_rural_male)

# SHAP 分析：以rural_male_lgbm为例
rural_male_lgbm.fit(X_tree_rural_male_train, Y_rural_male_train)
X_train_rural_male_shap = X_tree_rural_male_train.copy()
plot_shap(rural_male_lgbm, X_train_rural_male_shap, X_train_rural_male_shap.columns.tolist(), target_rural_male, n_samples=500)

###########################
# 农村女性模型部分
###########################
target_rural_female = "Rural_Female_Model"

print("\n=== 使用贝叶斯优化训练农村女性模型 ===")
# 使用贝叶斯优化来优化RandomForest模型
print("优化 RandomForest 模型...")
rural_female_rf = optimize_rf(X_tree_rural_female_train, Y_rural_female_train, n_iter=10)

# SVM模型
rural_female_svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=42)

# 使用贝叶斯优化来优化XGBoost模型
print("优化 XGBoost 模型...")
rural_female_xgb = optimize_xgb(X_tree_rural_female_train, Y_rural_female_train, n_iter=10)

# 使用贝叶斯优化来优化GBDT模型
print("优化 GBDT 模型...")
rural_female_gbdt = optimize_gbdt(X_tree_rural_female_train_imputed, Y_rural_female_train, n_iter=10)

# 使用贝叶斯优化来优化LightGBM模型
print("优化 LightGBM 模型...")
rural_female_lgbm = optimize_lgbm(X_tree_rural_female_train, Y_rural_female_train, n_iter=10)

print("\n=== 农村女性模型优化完成，开始评估 ===")
rural_female_rf_metrics = evaluate_model(rural_female_rf, X_tree_rural_female_val, Y_rural_female_val)
rural_female_svm_metrics = evaluate_model(rural_female_svm.fit(X_non_tree_rural_female_train, Y_rural_female_train), X_non_tree_rural_female_val, Y_rural_female_val)
rural_female_xgb_metrics = evaluate_model(rural_female_xgb, X_tree_rural_female_val, Y_rural_female_val)
rural_female_gbdt_metrics = evaluate_model(rural_female_gbdt, X_tree_rural_female_val_imputed, Y_rural_female_val)
rural_female_lgbm_metrics = evaluate_model(rural_female_lgbm, X_tree_rural_female_val, Y_rural_female_val)

print("Rural Female RF Metrics:", rural_female_rf_metrics)
print("Rural Female SVM Metrics:", rural_female_svm_metrics)
print("Rural Female XGBoost Metrics:", rural_female_xgb_metrics)
print("Rural Female GBDT Metrics:", rural_female_gbdt_metrics)
print("Rural Female LightGBM Metrics:", rural_female_lgbm_metrics)

print("\n=== Cross-validation for 农村女性模型 ===")
rural_female_rf_cv_train, rural_female_rf_cv_val = cross_validate(rural_female_rf, X_tree_rural_female_train, Y_rural_female_train, X_tree_rural_female_val, Y_rural_female_val, n_splits=10)
rural_female_svm_cv_train, rural_female_svm_cv_val = cross_validate(rural_female_svm, X_non_tree_rural_female_train, Y_rural_female_train, X_non_tree_rural_female_val, Y_rural_female_val, n_splits=10)
rural_female_xgb_cv_train, rural_female_xgb_cv_val = cross_validate(rural_female_xgb, X_tree_rural_female_train, Y_rural_female_train, X_tree_rural_female_val, Y_rural_female_val, n_splits=10)
rural_female_gbdt_cv_train, rural_female_gbdt_cv_val = cross_validate(rural_female_gbdt, X_tree_rural_female_train_imputed, Y_rural_female_train, X_tree_rural_female_val_imputed, Y_rural_female_val, n_splits=10)
rural_female_lgbm_cv_train, rural_female_lgbm_cv_val = cross_validate(rural_female_lgbm, X_tree_rural_female_train, Y_rural_female_train, X_tree_rural_female_val, Y_rural_female_val, n_splits=10)

# 构建农村女性CNN模型
rural_female_cnn = CNN(input_dim=X_non_tree_rural_female_train.shape[1], num_classes=4)
print("\n=== Cross-validation for 农村女性CNN模型 ===")
rural_female_cnn_cv_train, rural_female_cnn_cv_val = cross_validate(rural_female_cnn, X_non_tree_rural_female_train, Y_rural_female_train, X_non_tree_rural_female_val, Y_rural_female_val, n_splits=10, epochs=200, batch_size=32)
print("农村女性CNN Cross-validation Training Metrics:", rural_female_cnn_cv_train)
print("农村女性CNN Cross-validation Validation Metrics:", rural_female_cnn_cv_val)

rural_female_cnn_train_losses, rural_female_cnn_val_losses = train_single_model(rural_female_cnn, X_non_tree_rural_female_train, Y_rural_female_train, 
                                                               X_non_tree_rural_female_val, Y_rural_female_val, epochs=200, batch_size=32, patience=10)
save_cnn_model(rural_female_cnn, 'rural_female_cnn_model.pth')
rural_female_cnn_metrics = evaluate_model(rural_female_cnn, X_non_tree_rural_female_test, Y_rural_female_test)
print("农村女性CNN Test Metrics:", rural_female_cnn_metrics)

# 绘制农村女性CNN收敛曲线
plot_cnn_convergence(rural_female_cnn_train_losses, rural_female_cnn_val_losses, target_rural_female)

# 绘制农村女性模型评估图
# 评估农村女性模型在训练集上的性能
rural_female_cnn.eval()
rural_female_rf.fit(X_tree_rural_female_train, Y_rural_female_train)
rural_female_svm.fit(X_non_tree_rural_female_train, Y_rural_female_train) 
rural_female_xgb.fit(X_tree_rural_female_train, Y_rural_female_train)
rural_female_gbdt.fit(X_tree_rural_female_train_imputed, Y_rural_female_train)
rural_female_lgbm.fit(X_tree_rural_female_train, Y_rural_female_train)

rural_female_cnn_train_metrics = evaluate_model(rural_female_cnn, X_non_tree_rural_female_train, Y_rural_female_train)
rural_female_rf_train_metrics = evaluate_model(rural_female_rf, X_tree_rural_female_train, Y_rural_female_train)
rural_female_svm_train_metrics = evaluate_model(rural_female_svm, X_non_tree_rural_female_train, Y_rural_female_train)
rural_female_xgb_train_metrics = evaluate_model(rural_female_xgb, X_tree_rural_female_train, Y_rural_female_train)
rural_female_gbdt_train_metrics = evaluate_model(rural_female_gbdt, X_tree_rural_female_train_imputed, Y_rural_female_train)
rural_female_lgbm_train_metrics = evaluate_model(rural_female_lgbm, X_tree_rural_female_train, Y_rural_female_train)

# 评估农村女性模型在测试集上的性能
rural_female_cnn_test_metrics = evaluate_model(rural_female_cnn, X_non_tree_rural_female_test, Y_rural_female_test)
rural_female_rf_test_metrics = evaluate_model(rural_female_rf, X_tree_rural_female_test, Y_rural_female_test)
rural_female_svm_test_metrics = evaluate_model(rural_female_svm, X_non_tree_rural_female_test, Y_rural_female_test)
rural_female_xgb_test_metrics = evaluate_model(rural_female_xgb, X_tree_rural_female_test, Y_rural_female_test)
rural_female_gbdt_test_metrics = evaluate_model(rural_female_gbdt, X_tree_rural_female_test_imputed, Y_rural_female_test)
rural_female_lgbm_test_metrics = evaluate_model(rural_female_lgbm, X_tree_rural_female_test, Y_rural_female_test)

# 创建训练集和测试集性能指标列表
rural_female_model_train_metrics = [rural_female_cnn_train_metrics, rural_female_rf_train_metrics, rural_female_svm_train_metrics, rural_female_xgb_train_metrics, rural_female_gbdt_train_metrics, rural_female_lgbm_train_metrics]
rural_female_model_test_metrics = [rural_female_cnn_test_metrics, rural_female_rf_test_metrics, rural_female_svm_test_metrics, rural_female_xgb_test_metrics, rural_female_gbdt_test_metrics, rural_female_lgbm_test_metrics]

# 绘制农村女性模型评估图 - 使用区分的训练集和测试集指标
plot_model_evaluation(rural_female_model_train_metrics, rural_female_model_test_metrics, target_rural_female)


# 生成模型在测试集上的预测
# CNN预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_rural_female_tensor = torch.FloatTensor(X_non_tree_rural_female_test.values if hasattr(X_non_tree_rural_female_test, 'values') else X_non_tree_rural_female_test).to(device)
rural_female_cnn.eval()
with torch.no_grad():
    outputs = rural_female_cnn(X_rural_female_tensor)
    rural_female_cnn_preds = torch.max(outputs, 1)[1].cpu().numpy()
    rural_female_cnn_probas = torch.softmax(outputs, dim=1).cpu().numpy()

# 其他模型的预测
rural_female_rf.fit(X_tree_rural_female_train, Y_rural_female_train)
rural_female_rf_preds = rural_female_rf.predict(X_tree_rural_female_test)
rural_female_rf_probas = rural_female_rf.predict_proba(X_tree_rural_female_test)

rural_female_svm.fit(X_non_tree_rural_female_train, Y_rural_female_train)
rural_female_svm_preds = rural_female_svm.predict(X_non_tree_rural_female_test)
rural_female_svm_probas = rural_female_svm.predict_proba(X_non_tree_rural_female_test)

rural_female_xgb.fit(X_tree_rural_female_train, Y_rural_female_train)
rural_female_xgb_preds = rural_female_xgb.predict(X_tree_rural_female_test)
rural_female_xgb_probas = rural_female_xgb.predict_proba(X_tree_rural_female_test)

rural_female_gbdt.fit(X_tree_rural_female_train_imputed, Y_rural_female_train)
rural_female_gbdt_preds = rural_female_gbdt.predict(X_tree_rural_female_test_imputed)
rural_female_gbdt_probas = rural_female_gbdt.predict_proba(X_tree_rural_female_test_imputed)

rural_female_lgbm.fit(X_tree_rural_female_train, Y_rural_female_train)
rural_female_lgbm_preds = rural_female_lgbm.predict(X_tree_rural_female_test)
rural_female_lgbm_probas = rural_female_lgbm.predict_proba(X_tree_rural_female_test)

# 存储预测结果
all_predictions['Rural_Female']['CNN'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_cnn_preds}
all_predictions['Rural_Female']['RF'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_rf_preds}
all_predictions['Rural_Female']['SVM'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_svm_preds}
all_predictions['Rural_Female']['XGBoost'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_xgb_preds}
all_predictions['Rural_Female']['GBDT'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_gbdt_preds}
all_predictions['Rural_Female']['LightGBM'] = {'y_true': Y_rural_female_test, 'y_pred': rural_female_lgbm_preds}

# 绘制农村女性模型混淆矩阵（单个模型）
plot_all_confusion_matrices(Y_rural_female_test, {"Rural_Female_LightGBM": rural_female_lgbm_preds}, target_rural_female)

# 绘制农村女性各模型的ROC曲线
roc_dict_rural_female = {
    "Rural Female RF": rural_female_rf_probas,
    "Rural Female XGBoost": rural_female_xgb_probas,
    "Rural Female GBDT": rural_female_gbdt_probas,
    "Rural Female LightGBM": rural_female_lgbm_probas,
    "Rural Female SVM": rural_female_svm_probas,
    "Rural Female CNN": rural_female_cnn_probas
}
plot_roc_curves(Y_rural_female_test, roc_dict_rural_female, target_rural_female)

# SHAP 分析：以rural_female_lgbm为例
rural_female_lgbm.fit(X_tree_rural_female_train, Y_rural_female_train)
X_train_rural_female_shap = X_tree_rural_female_train.copy()
plot_shap(rural_female_lgbm, X_train_rural_female_shap, X_train_rural_female_shap.columns.tolist(), target_rural_female, n_samples=500)

# 保存最终模型和特征变量
with open('urban_male_lgbm_model.pkl', 'wb') as f:
    pickle.dump(urban_male_lgbm, f)
X_urban_male.to_csv('urban_male_features.csv', index=False)

with open('urban_female_lgbm_model.pkl', 'wb') as f:
    pickle.dump(urban_female_lgbm, f)
X_urban_female.to_csv('urban_female_features.csv', index=False)

with open('rural_male_lgbm_model.pkl', 'wb') as f:
    pickle.dump(rural_male_lgbm, f)
X_rural_male.to_csv('rural_male_features.csv', index=False)

with open('rural_female_lgbm_model.pkl', 'wb') as f:
    pickle.dump(rural_female_lgbm, f)
X_rural_female.to_csv('rural_female_features.csv', index=False)

# 保存编码器和缩放器
with open('category_mappings.pkl', 'wb') as f:
    pickle.dump(category_mappings, f)
with open('scaler_non_tree_urban_male.pkl', 'wb') as f:
    pickle.dump(scaler_non_tree_urban_male, f)
with open('scaler_non_tree_urban_female.pkl', 'wb') as f:
    pickle.dump(scaler_non_tree_urban_female, f)
with open('scaler_non_tree_rural_male.pkl', 'wb') as f:
    pickle.dump(scaler_non_tree_rural_male, f)
with open('scaler_non_tree_rural_female.pkl', 'wb') as f:
    pickle.dump(scaler_non_tree_rural_female, f)

# 绘制所有人群和模型的混淆矩阵：4行6列的大图
print("\n=== 绘制所有人群和模型的混淆矩阵图 ===")
population_names = ['Urban_Male', 'Urban_Female', 'Rural_Male', 'Rural_Female']
model_names = ['CNN', 'RF', 'SVM', 'XGBoost', 'GBDT', 'LightGBM']
plot_all_population_model_confusion_matrices(all_predictions, model_names, population_names)

print("城市男性、城市女性、农村男性、农村女性模型训练和评估完成。")