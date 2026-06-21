#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无泄漏版 —— 完整产物脚本 (p08)
==================================================================
在 p07 无泄漏特征集(土壤+气候+品种+人群, 剔除 蔬菜镉/BCF/SCC)的基础上,
一次性产出原脚本 p06 的全部图表与数据文件, 全部写入 ./noleak_outputs/ :

  数据导出:
    - 无泄漏数据集.xlsx                 (4人群sheet: 诚实特征+风险标签+train/val/test划分)
  每人群:
    - {pop}_cnn_convergence.pdf + _data.xlsx      CNN收敛曲线
    - {pop}_model_evaluation.pdf  + _data.xlsx     6模型 训练/测试 AUC/ACC/SE/F1 柱状图
    - {pop}_roc_curves.pdf        + _data.xlsx     6模型 ROC
    - {pop}_{model}_confusion_matrix.pdf           每模型混淆矩阵图
    - {pop}_confusion_matrix_{model}.xlsx          每模型混淆矩阵表格
    - {pop}_shap_*.pdf + _data.xlsx                LightGBM SHAP 全套
  汇总:
    - all_population_model_confusion_matrices.pdf/.png   4x6 大图
    - all_confusion_matrices_data.xlsx / _pivot.xlsx     全部混淆矩阵长表/透视表
    - test_acc_auc_table_NOLEAK.xlsx                     诚实ACC/AUC表 + 泄漏对比
==================================================================
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import copy
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
import xgboost as xgb
from lightgbm import LGBMClassifier
import shap
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

plt.rcParams['font.family'] = 'Times New Roman'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_STATE = 42
DATA_PATH = r'C:\Users\Administrator\Desktop\AUC\EN中国蔬菜镉含量数据库_all9_noleak.xlsx'
OUT = 'noleak_outputs'
os.makedirs(OUT, exist_ok=True)
op = lambda fn: os.path.join(OUT, fn)

# ==================================================================
# 1. 数据加载 / 目标 / 编码
# ==================================================================
raw = pd.read_excel(DATA_PATH)
raw.columns = raw.columns.str.strip().str.replace('\xa0', ' ', regex=False)

def map_thq_to_risk(thq):
    if thq <= 0.5: return 0
    elif thq <= 1: return 1
    elif thq <= 2: return 2
    else: return 3

raw['Urban Risk (Male)']   = raw['Urban THQ (Male)'].apply(map_thq_to_risk)
raw['Urban Risk (Female)'] = raw['Urban THQ (Female)'].apply(map_thq_to_risk)
raw['Rural Risk (Male)']   = raw['Rural THQ (Male)'].apply(map_thq_to_risk)
raw['Rural Risk (Female)'] = raw['Rural THQ (Female)'].apply(map_thq_to_risk)

# 编码后的数据(供建模); 同时保留原始未编码数据(供数据导出更可读)
data = raw.copy()
category_mappings = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].fillna('Missing')
    data[column] = le.fit_transform(data[column].astype(str)).astype('float64')
    category_mappings[column] = le

HONEST_COMMON = ['Region', 'Province', 'Climate Zone', 'Major Veg Category', 'Specific Veg Type',
                 'Soil Cadmium (mg/kg)', 'pH', 'SOM', 'CEC', 'Season', 'Year']
LEAKAGE_DROPPED = ['Vegetable Cadmium (mg/kg)', 'BCF', 'SCC']

POP_CONFIG = {
    'Urban_Male':   {'target': 'Urban Risk (Male)',
                     'pop_feats': ['Urban Veg Consumption (kg/year/capita)', 'Urban Body Weight (Male)']},
    'Urban_Female': {'target': 'Urban Risk (Female)',
                     'pop_feats': ['Urban Veg Consumption (kg/year/capita)', 'Urban Body Weight (Female)']},
    'Rural_Male':   {'target': 'Rural Risk (Male)',
                     'pop_feats': ['Rural Veg Consumption (kg/year/capita)', 'Rural Body Weight (Male)']},
    'Rural_Female': {'target': 'Rural Risk (Female)',
                     'pop_feats': ['Rural Veg Consumption (kg/year/capita)', 'Rural Body Weight (Female)']},
}
MODEL_NAMES = ['CNN', 'RF', 'SVM', 'XGBoost', 'GBDT', 'LightGBM']
METRICS = ['AUC', 'ACC', 'SE', 'F1']
POP_NAMES = list(POP_CONFIG.keys())

# ==================================================================
# 2. 模型 / 评估 / 优化
# ==================================================================
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2); self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2); self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 5, padding=2); self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256); self.fc2 = nn.Linear(256, 128); self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3); self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(1)
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x).view(x.size(0), -1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x)

def train_cnn(model, Xtr, Ytr, Xva, Yva, epochs=200, batch_size=32, patience=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    tt = lambda a: torch.FloatTensor(a.values if hasattr(a, 'values') else a)
    Xtr, Ytr = tt(Xtr).to(device), torch.LongTensor(Ytr).to(device)
    Xva, Yva = tt(Xva).to(device), torch.LongTensor(Yva).to(device)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    best, best_state, wait = float('inf'), None, 0
    tr_losses, va_losses = [], []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step(); tot += loss.item()
        tr_losses.append(tot / len(loader))
        model.eval()
        with torch.no_grad(): vl = crit(model(Xva), Yva).item()
        va_losses.append(vl)
        if vl < best: best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: model.load_state_dict(best_state); break
    if best_state is not None: model.load_state_dict(best_state)
    return tr_losses, va_losses

def predict_any(model, X):
    if isinstance(X, pd.DataFrame): X = X.values
    if hasattr(model, 'parameters'):
        device = next(model.parameters()).device; model.eval()
        with torch.no_grad():
            out = model(torch.FloatTensor(X).to(device))
            probas = torch.softmax(out, dim=1).cpu().numpy(); preds = probas.argmax(1)
    else:
        preds = model.predict(X); probas = model.predict_proba(X)
    return preds, probas

def metrics_from(y, preds, probas):
    try: a = roc_auc_score(y, probas, multi_class='ovr', average='macro', labels=[0,1,2,3])
    except Exception: a = np.nan
    return {'AUC': a, 'ACC': accuracy_score(y, preds),
            'SE': recall_score(y, preds, average='macro', zero_division=0),
            'F1': f1_score(y, preds, average='macro', zero_division=0)}

def _bayes(make, space, X, y, Xv, yv, n_calls=10):
    # 关键修复: 在【验证集】上打分选超参, 避免在训练集上fit又打分导致的过拟合
    @use_named_args(space)
    def obj(**p):
        m = make(p); m.fit(X, y); return -accuracy_score(yv, m.predict(Xv))
    res = gp_minimize(obj, space, n_calls=max(10, n_calls), random_state=RANDOM_STATE)
    best = {space[i].name: res.x[i] for i in range(len(space))}
    m = make(best); m.fit(X, y); return m

def optimize_rf(X, y, Xv, yv):
    s = [Integer(50,300,name='n_estimators'), Integer(5,30,name='max_depth'),
         Integer(2,20,name='min_samples_split'), Integer(1,10,name='min_samples_leaf'),
         Categorical(['sqrt','log2',None],name='max_features')]
    return _bayes(lambda p: RandomForestClassifier(random_state=RANDOM_STATE, **p), s, X, y, Xv, yv)
def optimize_xgb(X, y, Xv, yv):
    s = [Integer(50,300,name='n_estimators'), Real(0.01,0.3,'log-uniform',name='learning_rate'),
         Integer(3,10,name='max_depth'), Real(0.5,1.0,name='subsample'),
         Real(0.5,1.0,name='colsample_bytree'), Real(0,5,name='gamma'), Integer(1,10,name='min_child_weight')]
    return _bayes(lambda p: xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False,
                  eval_metric='mlogloss', objective='multi:softprob', num_class=4, **p), s, X, y, Xv, yv)
def optimize_gbdt(X, y, Xv, yv):
    s = [Integer(50,300,name='n_estimators'), Real(0.01,0.3,'log-uniform',name='learning_rate'),
         Integer(3,10,name='max_depth'), Integer(2,20,name='min_samples_split'),
         Integer(1,10,name='min_samples_leaf'), Real(0.5,1.0,name='subsample'),
         Categorical(['sqrt','log2',None],name='max_features')]
    return _bayes(lambda p: GradientBoostingClassifier(random_state=RANDOM_STATE, **p), s, X, y, Xv, yv)
def optimize_lgbm(X, y, Xv, yv):
    s = [Integer(50,300,name='n_estimators'), Real(0.01,0.3,'log-uniform',name='learning_rate'),
         Integer(20,100,name='num_leaves'), Integer(3,10,name='max_depth'),
         Integer(10,50,name='min_child_samples'), Real(0.5,1.0,name='subsample'),
         Real(0.5,1.0,name='colsample_bytree')]
    return _bayes(lambda p: LGBMClassifier(random_state=RANDOM_STATE, objective='multiclass',
                  num_class=4, verbose=-1, **p), s, X, y, Xv, yv)

# ==================================================================
# 3. 绘图函数 (移植自原脚本, 输出路径改到 OUT)
# ==================================================================
def plot_cnn_convergence(tr, va, pop):
    plt.figure(figsize=(10, 6))
    plt.plot(tr, label='Training Loss'); plt.plot(va, label='Validation Loss')
    plt.title(f'{pop} CNN Convergence Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=15); plt.ylabel('Loss', fontsize=15)
    plt.legend(prop={'size': 15}); plt.tick_params(labelsize=15); plt.tight_layout()
    plt.savefig(op(f'{pop}_cnn_convergence.pdf'), dpi=300, bbox_inches='tight'); plt.close()
    pd.DataFrame({'Epoch': range(1, len(tr)+1), 'Train Loss': tr, 'Validation Loss': va}).to_excel(
        op(f'{pop}_cnn_convergence_data.xlsx'), index=False)

def plot_model_evaluation(train_metrics, test_metrics, pop):
    colors = {'CNN':'#004775','RF':'#41A0BC','SVM':'#E45C5E','XGBoost':'#8E44AD','GBDT':'#27AE60','LightGBM':'#F39C12'}
    disp = ['AUC', 'ACC', 'SE', 'F1 score']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7)); bw = 0.13
    for i, m in enumerate(MODEL_NAMES):
        ax1.bar(np.arange(4) + i*bw, [train_metrics[m].get(k,0) for k in METRICS], width=bw, label=m, color=colors[m])
    ax1.set_title(f'{pop} Model Performance on Training Set', fontsize=16)
    ax1.set_xticks(np.arange(4) + 0.33); ax1.set_xticklabels(disp, fontsize=18); ax1.set_ylim(0,1.1)
    ax1.legend(prop={'size': 12}, loc='lower right'); ax1.set_ylabel('Score', fontsize=18)
    for i, m in enumerate(MODEL_NAMES):
        ax2.bar(np.arange(4) + i*bw, [test_metrics[m].get(k,0) for k in METRICS], width=bw, label=m, color=colors[m])
    ax2.set_title(f'{pop} Model Performance on Test Set', fontsize=16)
    ax2.set_xticks(np.arange(4) + 0.33); ax2.set_xticklabels(disp, fontsize=18); ax2.set_ylim(0,1.1)
    ax2.legend(prop={'size': 12}, loc='lower right'); ax2.set_ylabel('Score', fontsize=18)
    for ax in (ax1, ax2): ax.tick_params(labelsize=18)
    plt.tight_layout(); plt.savefig(op(f'{pop}_model_evaluation.pdf'), dpi=300, bbox_inches='tight'); plt.close()
    rows = []
    for m in MODEL_NAMES:
        for k, d in zip(METRICS, disp):
            rows.append({'Model': m, 'Metric': d, 'Train': train_metrics[m].get(k,0), 'Test': test_metrics[m].get(k,0)})
    pd.DataFrame(rows).to_excel(op(f'{pop}_model_evaluation_data.xlsx'), index=False)

def plot_roc_curves(y_true, proba_dict, pop):
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'])
    yb = label_binarize(y_true, classes=[0,1,2,3])
    for (name, proba), c in zip(proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(yb.ravel(), proba.ravel()); ra = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {ra:.2f})', color=c, lw=2)
    plt.plot([0,1],[0,1],'k--', lw=2); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{pop} ROC Curves', fontsize=14); plt.legend(loc='lower right', prop={'size': 10}); plt.tight_layout()
    plt.savefig(op(f'{pop}_roc_curves.pdf'), dpi=300); plt.close()
    with pd.ExcelWriter(op(f'{pop}_roc_curves_data.xlsx')) as w:
        for name, proba in proba_dict.items():
            fpr, tpr, _ = roc_curve(yb.ravel(), proba.ravel())
            pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(w, sheet_name=name[:31], index=False)

def plot_one_confusion(y_true, y_pred, pop, model):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{pop} {model}', fontsize=15); plt.xlabel('Predicted', fontsize=13); plt.ylabel('True', fontsize=13)
    plt.tight_layout(); plt.savefig(op(f'{pop}_{model}_confusion_matrix.pdf'), dpi=300, bbox_inches='tight'); plt.close()
    pd.DataFrame(cm, index=[f'True_{i}' for i in range(4)], columns=[f'Pred_{i}' for i in range(4)]).to_excel(
        op(f'{pop}_confusion_matrix_{model}.xlsx'))

def plot_shap_lgbm(model, X, feat_names, pop, n_samples=500):
    if len(X) > n_samples: X = X.sample(n=n_samples, random_state=RANDOM_STATE)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X); ev = explainer.expected_value
        if isinstance(sv, list): raw_sv = np.stack(sv, axis=2)
        else:
            raw_sv = sv if sv.ndim == 3 else np.expand_dims(sv, axis=2)
        base = np.array(ev) if isinstance(ev, (list, np.ndarray)) else np.array([ev])
        ncls = raw_sv.shape[2]
        for cl in range(ncls):
            csv = raw_sv[:, :, cl]
            expl = shap.Explanation(values=csv,
                base_values=np.full(len(X), base[cl] if len(base) > 1 else base[0]),
                data=X.values, feature_names=feat_names)
            for kind, fn in [('beeswarm', shap.plots.beeswarm), ('bar', shap.plots.bar)]:
                try:
                    plt.figure(figsize=(12, 8)); fn(expl, show=False, max_display=10)
                    plt.title(f'{pop} SHAP {kind} - Class {cl}', fontsize=16); plt.tight_layout()
                    plt.savefig(op(f'{pop}_shap_{kind}_class_{cl}.pdf'), dpi=300, bbox_inches='tight'); plt.close()
                except Exception as e: print(f'  shap {kind} cls{cl} 失败: {e}')
            pd.DataFrame({'Feature': feat_names, 'SHAP_Values': np.mean(np.abs(csv), axis=0)}
                         ).sort_values('SHAP_Values', ascending=False).to_excel(
                         op(f'{pop}_shap_bar_class_{cl}_data.xlsx'), index=False)
            try:
                plt.figure(figsize=(12, 8)); shap.plots.heatmap(expl, show=False, max_display=10)
                plt.title(f'{pop} SHAP heatmap - Class {cl}', fontsize=16); plt.tight_layout()
                plt.savefig(op(f'{pop}_shap_heatmap_class_{cl}.pdf'), dpi=300, bbox_inches='tight'); plt.close()
            except Exception as e: print(f'  shap heatmap cls{cl} 失败: {e}')
            sample_expl = shap.Explanation(values=raw_sv[0, :, cl],
                base_values=base[cl] if len(base) > 1 else base[0],
                data=X.iloc[0, :].values, feature_names=feat_names)
            try:
                plt.figure(figsize=(12, 8)); shap.plots.waterfall(sample_expl, show=False)
                plt.title(f'{pop} SHAP waterfall - Class {cl} (Sample 0)', fontsize=16); plt.tight_layout()
                plt.savefig(op(f'{pop}_shap_waterfall_class_{cl}.pdf'), dpi=300, bbox_inches='tight'); plt.close()
            except Exception as e: print(f'  shap waterfall cls{cl} 失败: {e}')
        # 额外: 对 class_idx 做 force / decision / dependence
        ci = min(2, ncls-1)
        try:
            plt.figure(figsize=(12, 3))
            shap.force_plot(base[ci] if len(base) > 1 else base[0], raw_sv[0, :, ci], X.iloc[0, :],
                            show=False, matplotlib=True)
            plt.title(f'{pop} SHAP Force (Sample 0, class {ci})', fontsize=16); plt.tight_layout()
            plt.savefig(op(f'{pop}_shap_force.pdf'), dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e: print(f'  shap force 失败: {e}')
        try:
            plt.figure(figsize=(12, 8))
            shap.decision_plot(base[ci] if len(base) > 1 else base[0], raw_sv[:, :, ci], X,
                               feature_names=feat_names, show=False)
            plt.title(f'{pop} SHAP Decision (class {ci})', fontsize=16); plt.tight_layout()
            plt.savefig(op(f'{pop}_shap_decision.pdf'), dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e: print(f'  shap decision 失败: {e}')
        try:
            fi = np.mean(np.abs(raw_sv[:, :, ci]), axis=0); top = feat_names[int(np.argmax(fi))]
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(top, raw_sv[:, :, ci], X, show=False)
            plt.title(f'{pop} SHAP Dependence ({top}, class {ci})', fontsize=16); plt.tight_layout()
            plt.savefig(op(f'{pop}_shap_dependence.pdf'), dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e: print(f'  shap dependence 失败: {e}')
    except Exception as e:
        print(f'  {pop} SHAP 整体失败: {e}')

def visualize_cnn_structure(model, input_dim, pop):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(op(f'{pop}_cnn_structure.txt'), 'w') as f:
        f.write(str(model.cpu()))
    try:
        from torchviz import make_dot
        x = torch.randn(1, input_dim).requires_grad_(True)
        y = model.cpu()(x)
        dot = make_dot(y, params=dict(list(model.cpu().named_parameters()) + [('x', x)]))
        dot.render(op(f'{pop}_cnn_structure'), format='png', cleanup=True)
    except Exception as e:
        print(f'  {pop} CNN结构图生成跳过(需Graphviz): {e}')

# ==================================================================
# 4. 主循环
# ==================================================================
all_predictions = {p: {} for p in POP_NAMES}
test_rows = []
data_export = {}   # pop -> DataFrame(原始可读特征 + 标签 + split)

for pop in POP_NAMES:
    cfg = POP_CONFIG[pop]; feats = HONEST_COMMON + cfg['pop_feats']
    X = data[feats].copy(); Y = data[cfg['target']].values
    print(f"\n{'='*60}\n人群: {pop} (特征数={len(feats)})\n{'='*60}")

    idx = np.arange(len(X))
    Xtv, Xte, Ytv, Yte, itv, ite = train_test_split(X, Y, idx, test_size=0.15, random_state=RANDOM_STATE)
    Xtr, Xva, Ytr, Yva, itr, iva = train_test_split(Xtv, Ytv, itv, test_size=0.1765, random_state=RANDOM_STATE)

    # 记录数据导出(用原始未编码值, 更可读)
    split_label = pd.Series('train', index=idx)
    split_label.loc[iva] = 'val'; split_label.loc[ite] = 'test'
    exp = raw.loc[X.index, feats].copy()
    exp['Risk_Label'] = Y
    exp['Split'] = split_label.values
    data_export[pop] = exp.reset_index(drop=True)

    # 树模型: imputer 仅训练集 fit
    imp = SimpleImputer(strategy='mean'); cols = Xtr.columns
    Xtr_i = pd.DataFrame(imp.fit_transform(Xtr), columns=cols, index=Xtr.index)
    Xva_i = pd.DataFrame(imp.transform(Xva), columns=cols, index=Xva.index)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=cols, index=Xte.index)

    # 非树模型: 缺失指示+特殊值+标准化, scaler 仅训练集 fit
    SP = -999
    def mk(df, ref=None):
        d = df.copy()
        for c in df.columns:
            d[f'{c}_missing'] = d[c].isnull().astype(int); d[c] = d[c].fillna(SP)
        if ref is not None: d = d.reindex(columns=ref, fill_value=0)
        return d
    Xtr_n = mk(Xtr); nc = Xtr_n.columns; Xva_n = mk(Xva, nc); Xte_n = mk(Xte, nc)
    sc = StandardScaler()
    Xtr_n = pd.DataFrame(sc.fit_transform(Xtr_n), columns=nc, index=Xtr.index)
    Xva_n = pd.DataFrame(sc.transform(Xva_n), columns=nc, index=Xva.index)
    Xte_n = pd.DataFrame(sc.transform(Xte_n), columns=nc, index=Xte.index)

    print("  优化 RF/XGB/GBDT/LightGBM ...")
    rf = optimize_rf(Xtr_i, Ytr, Xva_i, Yva); xgbm = optimize_xgb(Xtr_i, Ytr, Xva_i, Yva)
    gbdt = optimize_gbdt(Xtr_i, Ytr, Xva_i, Yva); lgbm = optimize_lgbm(Xtr_i, Ytr, Xva_i, Yva)
    print("  训练 SVM ...")
    svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=RANDOM_STATE).fit(Xtr_n, Ytr)
    print("  训练 CNN ...")
    cnn = CNN(input_dim=Xtr_n.shape[1], num_classes=4)
    tr_losses, va_losses = train_cnn(cnn, Xtr_n, Ytr, Xva_n, Yva, epochs=200, batch_size=32, patience=10)

    # 训练集 / 测试集 数据映射
    train_io = {'CNN': (cnn, Xtr_n), 'RF': (rf, Xtr_i), 'SVM': (svm, Xtr_n),
                'XGBoost': (xgbm, Xtr_i), 'GBDT': (gbdt, Xtr_i), 'LightGBM': (lgbm, Xtr_i)}
    test_io = {'CNN': (cnn, Xte_n), 'RF': (rf, Xte_i), 'SVM': (svm, Xte_n),
               'XGBoost': (xgbm, Xte_i), 'GBDT': (gbdt, Xte_i), 'LightGBM': (lgbm, Xte_i)}

    train_metrics, test_metrics, proba_dict = {}, {}, {}
    for m in MODEL_NAMES:
        mdl, Xt = train_io[m]; p_tr, pr_tr = predict_any(mdl, Xt)
        train_metrics[m] = metrics_from(Ytr, p_tr, pr_tr)
        mdl, Xt = test_io[m]; p_te, pr_te = predict_any(mdl, Xt)
        test_metrics[m] = metrics_from(Yte, p_te, pr_te)
        proba_dict[f'{pop} {m}'] = pr_te
        all_predictions[pop][m] = {'y_true': Yte, 'y_pred': p_te, 'y_proba': pr_te}
        test_rows.append({'Population': pop, 'Model': m, **test_metrics[m]})
        plot_one_confusion(Yte, p_te, pop, m)
        print(f"    {m:9s} test ACC={test_metrics[m]['ACC']:.4f} AUC={test_metrics[m]['AUC']:.4f}")

    # 图表
    plot_cnn_convergence(tr_losses, va_losses, pop)
    plot_model_evaluation(train_metrics, test_metrics, pop)
    plot_roc_curves(Yte, proba_dict, pop)
    visualize_cnn_structure(cnn, Xtr_n.shape[1], pop)
    print("  SHAP (LightGBM) ...")
    plot_shap_lgbm(lgbm, Xtr_i.copy(), list(cols), pop, n_samples=500)

# ==================================================================
# 5. 汇总: ACC/AUC 表 + 泄漏对比 + 4x6 混淆矩阵 + 全混淆矩阵表
# ==================================================================
df_long = pd.DataFrame(test_rows)
df_acc = df_long.pivot(index='Population', columns='Model', values='ACC').reindex(index=POP_NAMES, columns=MODEL_NAMES)
df_auc = df_long.pivot(index='Population', columns='Model', values='AUC').reindex(index=POP_NAMES, columns=MODEL_NAMES)

LEAK_ACC = {'Urban_Male':{'CNN':0.7488,'RF':0.9602,'SVM':0.7015,'XGBoost':0.9677,'GBDT':0.9577,'LightGBM':0.9677},
 'Urban_Female':{'CNN':0.7886,'RF':0.9602,'SVM':0.6915,'XGBoost':0.9602,'GBDT':0.9652,'LightGBM':0.9701},
 'Rural_Male':{'CNN':0.7836,'RF':0.9577,'SVM':0.7015,'XGBoost':0.9751,'GBDT':0.9627,'LightGBM':0.9726},
 'Rural_Female':{'CNN':0.7811,'RF':0.9577,'SVM':0.6866,'XGBoost':0.9726,'GBDT':0.9677,'LightGBM':0.9776}}
LEAK_AUC = {'Urban_Male':{'CNN':0.9147,'RF':0.9954,'SVM':0.8614,'XGBoost':0.9966,'GBDT':0.9961,'LightGBM':0.9960},
 'Urban_Female':{'CNN':0.9308,'RF':0.9962,'SVM':0.8642,'XGBoost':0.9968,'GBDT':0.9971,'LightGBM':0.9979},
 'Rural_Male':{'CNN':0.9364,'RF':0.9964,'SVM':0.8613,'XGBoost':0.9976,'GBDT':0.9972,'LightGBM':0.9979},
 'Rural_Female':{'CNN':0.9248,'RF':0.9964,'SVM':0.8621,'XGBoost':0.9980,'GBDT':0.9977,'LightGBM':0.9993}}
cmp_rows = []
for pop in POP_NAMES:
    for m in MODEL_NAMES:
        ha, hu = df_acc.loc[pop, m], df_auc.loc[pop, m]
        cmp_rows.append({'Population': pop, 'Model': m, 'ACC_leak': LEAK_ACC[pop][m], 'ACC_honest': ha,
            'ACC_drop': LEAK_ACC[pop][m]-ha, 'AUC_leak': LEAK_AUC[pop][m], 'AUC_honest': hu,
            'AUC_drop': LEAK_AUC[pop][m]-hu})
df_cmp = pd.DataFrame(cmp_rows)

with pd.ExcelWriter(op('test_acc_auc_table_NOLEAK.xlsx')) as w:
    df_long.to_excel(w, sheet_name='honest_long', index=False)
    df_acc.to_excel(w, sheet_name='honest_ACC'); df_auc.to_excel(w, sheet_name='honest_AUC')
    df_cmp.to_excel(w, sheet_name='leak_vs_honest', index=False)
    pd.DataFrame({'dropped_leakage_features': LEAKAGE_DROPPED}).to_excel(w, sheet_name='dropped_features', index=False)

# 4x6 大图
fig = plt.figure(figsize=(26, 20)); gs = gridspec.GridSpec(4, 6, figure=fig, wspace=0.3, hspace=0.3)
cm_long = []
for i, pop in enumerate(POP_NAMES):
    for j, m in enumerate(MODEL_NAMES):
        e = all_predictions[pop][m]; cm = confusion_matrix(e['y_true'], e['y_pred'], labels=[0,1,2,3])
        for ii in range(4):
            for jj in range(4):
                cm_long.append({'Population': pop, 'Model': m, 'True_Label': ii, 'Predicted_Label': jj, 'Count': cm[ii, jj]})
        ax = fig.add_subplot(gs[i, j]); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, square=True)
        if i == 0: ax.set_title(m, fontsize=14)
        if j == 0: ax.set_ylabel(pop.replace('_', ' '), fontsize=14)
        ax.set_xlabel('Predicted' if i == len(POP_NAMES)-1 else '')
plt.suptitle('Confusion Matrices for All Populations and Models (No-Leakage)', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(op('all_population_model_confusion_matrices.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(op('all_population_model_confusion_matrices.png'), dpi=200, bbox_inches='tight'); plt.close()

df_cm = pd.DataFrame(cm_long)
df_cm.to_excel(op('all_confusion_matrices_data.xlsx'), index=False)
df_cm.pivot_table(index=['Population','True_Label'], columns=['Model','Predicted_Label'],
                  values='Count', aggfunc='sum').fillna(0).astype(int).to_excel(op('all_confusion_matrices_pivot.xlsx'))

# 无泄漏数据集导出
with pd.ExcelWriter(op('无泄漏数据集.xlsx')) as w:
    info = pd.DataFrame({
        '说明': ['本文件为剔除目标泄漏特征后的诚实建模数据集',
               f'保留的公共特征: {HONEST_COMMON}',
               '人群特有特征: 摄入量 + 体重 (各人群不同)',
               f'已剔除的泄漏特征: {LEAKAGE_DROPPED}',
               'Risk_Label: 0无风险 1低 2中 3高 (由THQ分箱, 仅作标签不入特征)',
               'Split: train/val/test (与建模一致, random_state=42, 15%测试)']})
    info.to_excel(w, sheet_name='README', index=False)
    for pop in POP_NAMES:
        data_export[pop].to_excel(w, sheet_name=pop, index=False)

# 控制台汇总
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
print("\n【无泄漏】测试集 ACC\n", df_acc)
print("\n【无泄漏】测试集 AUC\n", df_auc)
print(f"\n全部产物已写入: {os.path.abspath(OUT)}")
print("无泄漏版完整产物生成完毕。")
