#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无泄漏版 —— 完整产物脚本 (p08) - 输出所有风险等级结果
使用p06的颜色配置和图形大小
==================================================================
适配新数据集格式：
  - 数据从4个sheet读取（Urban_Male, Urban_Female, Rural_Male, Rural_Female）
  - 每个sheet已包含 Risk_Label 和 Split 列
  - 输出所有风险等级（class_0, class_1, class_2, class_3）的SHAP图形和Excel
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
from matplotlib.colors import LinearSegmentedColormap
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

# ============ p06样式配置 ============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

JOURNAL_FONT = 'Times New Roman'
MODEL_COLORS = {
    'LightGBM': '#2E86C1',
    'CNN': '#3498DB',
    'RF': '#5DADE2',
    'SVM': '#85C1E9',
    'XGBoost': '#A9CCE3',
    'GBDT': '#C9D6DF'
}
MODEL_EDGE_COLORS = {
    'LightGBM': '#1F618D',
    'CNN': '#2878B5',
    'RF': '#418FBE',
    'SVM': '#5FA6D1',
    'XGBoost': '#82AFCB',
    'GBDT': '#A8B8C2'
}
ROC_MODEL_COLORS = {
    'LightGBM': '#1A5F7A',
    'RF': '#F4A261',
    'XGBoost': '#3498DB',
    'GBDT': '#D9534F',
    'SVM': '#7D6B91',
    'CNN': '#E76F51'
}
SHAP_BEESWARM_CMAP = LinearSegmentedColormap.from_list(
    'nature_biotech_shap',
    ['#3498DB', '#E2E2E2', '#E74C3C'],
    N=256
)

def get_model_color(model_name):
    for key, color in MODEL_COLORS.items():
        if key.lower() in str(model_name).lower():
            return color
    return '#4D4D4D'

def get_model_edge_color(model_name):
    for key, color in MODEL_EDGE_COLORS.items():
        if key.lower() in str(model_name).lower():
            return color
    return '#3D3D3D'

def get_roc_model_color(model_name):
    for key, color in ROC_MODEL_COLORS.items():
        if key.lower() in str(model_name).lower():
            return color
    return '#4D4D4D'

def is_core_model(model_name):
    return 'lightgbm' in str(model_name).lower()

def apply_publication_axis_style(ax, grid=True):
    sns.despine(ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        length=3.5,
        width=0.8,
        labelsize=10,
        colors='#222222'
    )
    if grid:
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='#EEEEEE')
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

def restyle_shap_axes(fig=None, grid=False):
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        apply_publication_axis_style(ax, grid=grid)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.title.set_fontname(JOURNAL_FONT)
        ax.xaxis.label.set_fontname(JOURNAL_FONT)
        ax.yaxis.label.set_fontname(JOURNAL_FONT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(JOURNAL_FONT)

def set_centered_shap_xlim(ax, shap_values, lower_q=2, upper_q=98,
                           min_zero_pos=0.40, max_zero_pos=0.60):
    values = np.asarray(shap_values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    lo, hi = np.percentile(values, [lower_q, upper_q])
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        max_abs = float(np.percentile(np.abs(values), upper_q))
        if max_abs <= 0 or not np.isfinite(max_abs):
            max_abs = 1.0
        lo, hi = -max_abs, max_abs

    span = hi - lo
    pad = max(span * 0.06, 1e-6)
    lo -= pad
    hi += pad

    zero_pos = (0.0 - lo) / (hi - lo)
    if zero_pos < min_zero_pos and hi > 0:
        lo = min(lo, -(min_zero_pos * hi) / (1.0 - min_zero_pos))
    elif zero_pos > max_zero_pos and lo < 0:
        hi = max(hi, lo * (max_zero_pos - 1.0) / max_zero_pos)

    ax.set_xlim(lo, hi)
    ax.axvline(0, color='#6A6A6A', linewidth=0.75, alpha=0.75, zorder=0)

# ============ 配置 ============
RANDOM_STATE = 42
DATA_PATH = './EN中国蔬菜镉含量数据库_all9_noleak_remaining2232_补充2022-2026.xlsx'
OUT = 'noleak_outputs_all_classes'
os.makedirs(OUT, exist_ok=True)
op = lambda fn: os.path.join(OUT, fn)

MODEL_NAMES = ['CNN', 'RF', 'SVM', 'XGBoost', 'GBDT', 'LightGBM']
METRICS = ['AUC', 'ACC', 'SE', 'F1']
POP_NAMES = ['Urban_Male', 'Urban_Female', 'Rural_Male', 'Rural_Female']

# ==================================================================
# CNN模型
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
# 绘图函数 - 使用p06样式，仅输出class_2
# ==================================================================
def plot_cnn_convergence(tr, va, pop):
    """CNN收敛曲线 - p06样式"""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(tr, label='Training Loss', color='#3498DB', linewidth=2.0)
    ax.plot(va, label='Validation Loss', color='#E74C3C', linewidth=2.0)
    ax.set_title(f'{pop} CNN convergence curve', fontsize=12, fontname=JOURNAL_FONT, pad=8)
    ax.set_xlabel('Epochs', fontsize=11, fontname=JOURNAL_FONT)
    ax.set_ylabel('Loss', fontsize=11, fontname=JOURNAL_FONT)
    apply_publication_axis_style(ax, grid=True)
    ax.legend(frameon=False, prop={'family': JOURNAL_FONT, 'size': 9}, loc='best')
    fig.tight_layout()
    fig.savefig(op(f'{pop}_cnn_convergence.pdf'), dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'Epoch': range(1, len(tr)+1), 'Train Loss': tr, 'Validation Loss': va}).to_excel(
        op(f'{pop}_cnn_convergence_data.xlsx'), index=False)

def plot_model_evaluation(train_metrics, val_metrics, pop):
    """模型评估对比图 - p06样式"""
    models = MODEL_NAMES
    metrics = METRICS
    display_names = ['AUC', 'ACC', 'SE', 'F1 score']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    bar_width = 0.13
    x = np.arange(len(metrics))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_width

    for ax, metric_source, panel_title in [
        (ax1, train_metrics, 'Training set (10-fold CV)'),
        (ax2, val_metrics, 'Test set')
    ]:
        for i, model in enumerate(models):
            if model in metric_source and metric_source[model] is not None:
                values = [metric_source[model].get(metric, 0) for metric in metrics]
                ax.bar(
                    x + offsets[i],
                    values,
                    width=bar_width,
                    label=model,
                    color=get_model_color(model),
                    alpha=1.0 if is_core_model(model) else 0.88,
                    edgecolor=get_model_edge_color(model) if is_core_model(model) else '#FFFFFF',
                    linewidth=0.9 if is_core_model(model) else 0.4,
                    zorder=3 if is_core_model(model) else 2
                )
        ax.set_title(panel_title, fontsize=11, fontname=JOURNAL_FONT, pad=6)
        ax.set_ylabel('Score', fontsize=10.5, fontname=JOURNAL_FONT)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        apply_publication_axis_style(ax, grid=True)

    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, fontsize=10, fontname=JOURNAL_FONT)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=6,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
        prop={'family': JOURNAL_FONT, 'size': 8.5},
        handlelength=1.2,
        columnspacing=1.0
    )
    fig.suptitle(f'{pop} model performance', fontsize=12.5, fontname=JOURNAL_FONT, y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(op(f'{pop}_model_evaluation.pdf'), dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)

    rows = []
    for model in models:
        if model in train_metrics and model in val_metrics:
            for metric, display_name in zip(metrics, display_names):
                rows.append({
                    'Model': model,
                    'Metric': display_name,
                    'Train': train_metrics[model].get(metric, 0),
                    'Test': val_metrics[model].get(metric, 0)
                })
    pd.DataFrame(rows).to_excel(op(f'{pop}_model_evaluation_data.xlsx'), index=False)

def plot_roc_curves(y_true, proba_dict, pop):
    """ROC曲线 - p06样式"""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    first_proba = np.asarray(next(iter(proba_dict.values())))
    classes = np.arange(first_proba.shape[1]) if first_proba.ndim > 1 else np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)

    roc_export = {}
    for name, y_pred_proba in proba_dict.items():
        y_pred_proba = np.asarray(y_pred_proba)
        per_class_curves = []
        all_fpr = []
        for class_idx, class_label in enumerate(classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, class_idx], y_pred_proba[:, class_idx])
            class_auc = auc(fpr_i, tpr_i)
            per_class_curves.append((class_label, fpr_i, tpr_i, class_auc))
            all_fpr.append(fpr_i)

        macro_fpr = np.unique(np.concatenate(all_fpr))
        mean_tpr = np.zeros_like(macro_fpr)
        for _, fpr_i, tpr_i, _ in per_class_curves:
            mean_tpr += np.interp(macro_fpr, fpr_i, tpr_i)
        mean_tpr /= len(per_class_curves)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        roc_auc = auc(macro_fpr, mean_tpr)

        model_name = name.replace(f'{pop} ', '')
        roc_export[model_name] = pd.DataFrame({'FPR': macro_fpr, 'TPR': mean_tpr})
        ax.plot(
            macro_fpr,
            mean_tpr,
            label=f'{model_name} (macro AUC = {roc_auc:.3f})',
            color=get_roc_model_color(model_name),
            linestyle='-',
            linewidth=2.2 if is_core_model(model_name) else 1.8,
            alpha=0.94 if is_core_model(model_name) else 0.86,
            zorder=3 if is_core_model(model_name) else 2
        )

    ax.plot([0, 1], [0, 1], color='#BDBDBD', linestyle='--', linewidth=0.9, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel('False positive rate', fontsize=10.5, fontname=JOURNAL_FONT)
    ax.set_ylabel('True positive rate', fontsize=10.5, fontname=JOURNAL_FONT)
    ax.set_title(f'{pop} ROC curves', fontsize=12, fontname=JOURNAL_FONT, pad=8)
    apply_publication_axis_style(ax, grid=True)
    ax.legend(
        loc='lower right',
        frameon=False,
        prop={'family': JOURNAL_FONT, 'size': 8.5},
        handlelength=1.8,
        borderaxespad=0.2
    )
    fig.tight_layout()
    fig.savefig(op(f'{pop}_roc_curves.pdf'), dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)

    with pd.ExcelWriter(op(f'{pop}_roc_curves_data.xlsx')) as writer:
        for name, df in roc_export.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)

def plot_one_confusion(y_true, y_pred, pop, model):
    """混淆矩阵 - p06样式"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=sns.light_palette('#3498DB', as_cmap=True),
        cbar=False,
        linewidths=0.4,
        linecolor='#FFFFFF',
        annot_kws={'fontsize': 10, 'fontname': JOURNAL_FONT},
        ax=ax
    )
    ax.set_title(f'{model}', fontname=JOURNAL_FONT, fontsize=12, pad=8)
    ax.set_xlabel('Predicted', fontname=JOURNAL_FONT, fontsize=10.5)
    ax.set_ylabel('True', fontname=JOURNAL_FONT, fontsize=10.5)
    apply_publication_axis_style(ax, grid=False)
    fig.tight_layout()
    fig.savefig(op(f'{pop}_{model}_confusion_matrices.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(cm).to_excel(op(f'{pop}_confusion_matrix_{model}.xlsx'), index=False)

def plot_shap_lgbm_all_classes(model, X, feat_names, pop, n_samples=500):
    """SHAP分析 - p06样式，输出所有风险等级"""
    print(f"  开始SHAP分析 (所有风险等级) for {pop}...")
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        ev = explainer.expected_value

        if isinstance(sv, list):
            raw_sv = np.stack(sv, axis=2)
        else:
            raw_sv = sv if sv.ndim == 3 else np.expand_dims(sv, axis=2)

        base = np.array(ev) if isinstance(ev, (list, np.ndarray)) else np.array([ev])
        ncls = raw_sv.shape[2]

        # 处理所有类别
        print(f"  生成 {ncls} 个风险等级的SHAP图...")
        for cl in range(ncls):
            print(f"    处理 class_{cl}...")

            csv = raw_sv[:, :, cl]
            base_val = base[cl] if len(base) > cl else base[0]
            expl = shap.Explanation(
                values=csv,
                base_values=np.full(len(X), base_val),
                data=X.values,
                feature_names=feat_names
            )

            # Beeswarm
            try:
                fig = plt.figure(figsize=(7.4, 5.6))
                shap.plots.beeswarm(expl, show=False, max_display=10)
                ax = plt.gca()
                ax.set_title(
                    f'{pop} SHAP beeswarm - class {cl}',
                    fontsize=12,
                    fontname=JOURNAL_FONT,
                    pad=8
                )
                ax.set_xlabel('SHAP value (impact on model output)', fontsize=10.5, fontname=JOURNAL_FONT)
                restyle_shap_axes(fig, grid=False)
                set_centered_shap_xlim(ax, csv)
                fig.tight_layout()
                fig.savefig(op(f'{pop}_shap_beeswarm_class_{cl}.pdf'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"      Beeswarm失败: {e}")

            # Bar
            try:
                fig = plt.figure(figsize=(7.2, 5.2))
                shap.plots.bar(expl, show=False, max_display=10)
                ax = plt.gca()
                for patch in ax.patches:
                    patch.set_facecolor('#2E86C1')
                    patch.set_edgecolor('#1F618D')
                    patch.set_linewidth(0.5)
                    patch.set_alpha(0.95)
                ax.set_title(
                    f'{pop} SHAP importance - class {cl}',
                    fontsize=12,
                    fontname=JOURNAL_FONT,
                    pad=8
                )
                restyle_shap_axes(fig, grid=True)
                fig.tight_layout()
                fig.savefig(op(f'{pop}_shap_bar_class_{cl}.pdf'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"      Bar失败: {e}")

            # 导出数据
            try:
                bar_data = pd.DataFrame({
                    'Feature': feat_names,
                    'SHAP_Values': np.mean(np.abs(csv), axis=0)
                }).sort_values('SHAP_Values', ascending=False)
                bar_data.to_excel(op(f'{pop}_shap_bar_class_{cl}_data.xlsx'), index=False)
            except Exception as e:
                print(f"      导出bar数据失败: {e}")

            # Heatmap
            try:
                fig = plt.figure(figsize=(7.4, 5.6))
                shap.plots.heatmap(expl, show=False, max_display=10)
                plt.gca().set_title(
                    f'{pop} SHAP heatmap - class {cl}',
                    fontsize=12,
                    fontname=JOURNAL_FONT,
                    pad=8
                )
                restyle_shap_axes(fig, grid=False)
                fig.tight_layout()
                fig.savefig(op(f'{pop}_shap_heatmap_class_{cl}.pdf'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"      Heatmap失败: {e}")

            # Waterfall
            try:
                sample_expl = shap.Explanation(
                    values=raw_sv[0, :, cl],
                    base_values=base_val,
                    data=X.iloc[0, :].values,
                    feature_names=feat_names
                )
                fig = plt.figure(figsize=(7.4, 5.2))
                shap.plots.waterfall(sample_expl, show=False)
                plt.gca().set_title(
                    f'{pop} SHAP waterfall - class {cl}',
                    fontsize=12,
                    fontname=JOURNAL_FONT,
                    pad=8
                )
                restyle_shap_axes(fig, grid=False)
                fig.tight_layout()
                fig.savefig(op(f'{pop}_shap_waterfall_class_{cl}.pdf'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"      Waterfall失败: {e}")

        # Force plot和Decision plot只生成最后一个类别的（因为它们是综合性的）
        ci = min(2, ncls-1)
        base_val_ci = base[ci] if len(base) > ci else base[0]

        # Force plot
        try:
            fig = plt.figure(figsize=(7.4, 2.4))
            shap.force_plot(
                base_val_ci,
                raw_sv[0, :, ci],
                X.iloc[0, :],
                show=False,
                matplotlib=True
            )
            plt.gca().set_title(
                f'{pop} SHAP force plot - class {ci}',
                fontsize=11.5,
                fontname=JOURNAL_FONT,
                pad=8
            )
            restyle_shap_axes(fig, grid=False)
            fig.tight_layout()
            fig.savefig(op(f'{pop}_shap_force.pdf'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"    Force失败: {e}")

        # Decision plot
        try:
            fig = plt.figure(figsize=(7.4, 5.6))
            shap.decision_plot(
                base_val_ci,
                raw_sv[:, :, ci],
                X,
                feature_names=feat_names,
                show=False
            )
            plt.gca().set_title(
                f'{pop} SHAP decision plot - class {ci}',
                fontsize=12,
                fontname=JOURNAL_FONT,
                pad=8
            )
            restyle_shap_axes(fig, grid=False)
            fig.tight_layout()
            fig.savefig(op(f'{pop}_shap_decision.pdf'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"    Decision失败: {e}")

    except Exception as e:
        print(f"  {pop} SHAP整体失败: {e}")



# ==================================================================
# 主循环
# ==================================================================
print(f"开始处理，输出目录: {os.path.abspath(OUT)}")
print("="*60)

all_predictions = {p: {} for p in POP_NAMES}
test_rows = []

for pop in POP_NAMES:
    print(f"\n{'='*60}\n处理人群: {pop}\n{'='*60}")

    # 从对应sheet读取数据
    df = pd.read_excel(DATA_PATH, sheet_name=pop)
    df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=False)

    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")

    # 提取特征和标签
    Y = df['Risk_Label'].values
    split_col = df['Split'].values

    # 特征列 = 除了 Risk_Label 和 Split 的所有列
    feature_cols = [c for c in df.columns if c not in ['Risk_Label', 'Split']]
    X_raw = df[feature_cols].copy()

    # 编码对象类型列
    X = X_raw.copy()
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing')
        X[col] = le.fit_transform(X[col].astype(str)).astype('float64')

    # 根据Split列划分数据
    train_mask = split_col == 'train'
    val_mask = split_col == 'val'
    test_mask = split_col == 'test'

    Xtr = X[train_mask].reset_index(drop=True)
    Ytr = Y[train_mask]
    Xva = X[val_mask].reset_index(drop=True)
    Yva = Y[val_mask]
    Xte = X[test_mask].reset_index(drop=True)
    Yte = Y[test_mask]

    print(f"  训练集: {len(Xtr)}, 验证集: {len(Xva)}, 测试集: {len(Xte)}")

    # 树模型数据：imputer
    imp = SimpleImputer(strategy='mean')
    cols = Xtr.columns
    Xtr_i = pd.DataFrame(imp.fit_transform(Xtr), columns=cols)
    Xva_i = pd.DataFrame(imp.transform(Xva), columns=cols)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=cols)

    # 非树模型数据：缺失指示 + 标准化
    SP = -999
    def mk(df, ref=None):
        d = df.copy()
        for c in df.columns:
            d[f'{c}_missing'] = d[c].isnull().astype(int)
            d[c] = d[c].fillna(SP)
        if ref is not None:
            d = d.reindex(columns=ref, fill_value=0)
        return d

    Xtr_n = mk(Xtr)
    nc = Xtr_n.columns
    Xva_n = mk(Xva, nc)
    Xte_n = mk(Xte, nc)

    sc = StandardScaler()
    Xtr_n = pd.DataFrame(sc.fit_transform(Xtr_n), columns=nc)
    Xva_n = pd.DataFrame(sc.transform(Xva_n), columns=nc)
    Xte_n = pd.DataFrame(sc.transform(Xte_n), columns=nc)

    # 训练模型
    print("  优化模型...")
    rf = optimize_rf(Xtr_i, Ytr, Xva_i, Yva)
    xgbm = optimize_xgb(Xtr_i, Ytr, Xva_i, Yva)
    gbdt = optimize_gbdt(Xtr_i, Ytr, Xva_i, Yva)
    lgbm = optimize_lgbm(Xtr_i, Ytr, Xva_i, Yva)

    print("  训练SVM...")
    svm = SVC(probability=True, kernel='rbf', decision_function_shape='ovr',
              random_state=RANDOM_STATE).fit(Xtr_n, Ytr)

    print("  训练CNN...")
    cnn = CNN(input_dim=Xtr_n.shape[1], num_classes=4)
    tr_losses, va_losses = train_cnn(cnn, Xtr_n, Ytr, Xva_n, Yva, epochs=200,
                                      batch_size=32, patience=10)

    # 评估
    train_io = {
        'CNN': (cnn, Xtr_n), 'RF': (rf, Xtr_i), 'SVM': (svm, Xtr_n),
        'XGBoost': (xgbm, Xtr_i), 'GBDT': (gbdt, Xtr_i), 'LightGBM': (lgbm, Xtr_i)
    }
    test_io = {
        'CNN': (cnn, Xte_n), 'RF': (rf, Xte_i), 'SVM': (svm, Xte_n),
        'XGBoost': (xgbm, Xte_i), 'GBDT': (gbdt, Xte_i), 'LightGBM': (lgbm, Xte_i)
    }

    train_metrics, test_metrics, proba_dict = {}, {}, {}
    for m in MODEL_NAMES:
        mdl, Xt = train_io[m]
        p_tr, pr_tr = predict_any(mdl, Xt)
        train_metrics[m] = metrics_from(Ytr, p_tr, pr_tr)

        mdl, Xt = test_io[m]
        p_te, pr_te = predict_any(mdl, Xt)
        test_metrics[m] = metrics_from(Yte, p_te, pr_te)
        proba_dict[f'{pop} {m}'] = pr_te
        all_predictions[pop][m] = {'y_true': Yte, 'y_pred': p_te, 'y_proba': pr_te}
        test_rows.append({'Population': pop, 'Model': m, **test_metrics[m]})

        plot_one_confusion(Yte, p_te, pop, m)
        print(f"    {m:9s} test ACC={test_metrics[m]['ACC']:.4f} AUC={test_metrics[m]['AUC']:.4f}")

    # 绘图
    plot_cnn_convergence(tr_losses, va_losses, pop)
    plot_model_evaluation(train_metrics, test_metrics, pop)
    plot_roc_curves(Yte, proba_dict, pop)

    # SHAP - 所有风险等级
    print("  生成SHAP图 (所有风险等级)...")
    plot_shap_lgbm_all_classes(lgbm, Xtr_i.copy(), list(cols), pop, n_samples=500)

# ==================================================================
# 汇总
# ==================================================================
print("\n" + "="*60)
print("生成汇总文件...")
print("="*60)

df_long = pd.DataFrame(test_rows)
df_acc = df_long.pivot(index='Population', columns='Model', values='ACC').reindex(
    index=POP_NAMES, columns=MODEL_NAMES)
df_auc = df_long.pivot(index='Population', columns='Model', values='AUC').reindex(
    index=POP_NAMES, columns=MODEL_NAMES)

with pd.ExcelWriter(op('test_acc_auc_table_all_classes.xlsx')) as w:
    df_long.to_excel(w, sheet_name='all_metrics', index=False)
    df_acc.to_excel(w, sheet_name='ACC')
    df_auc.to_excel(w, sheet_name='AUC')

# 4x6混淆矩阵大图
fig = plt.figure(figsize=(26, 20))
gs = gridspec.GridSpec(4, 6, figure=fig, wspace=0.3, hspace=0.3)
cm_long = []

for i, pop in enumerate(POP_NAMES):
    for j, m in enumerate(MODEL_NAMES):
        e = all_predictions[pop][m]
        cm = confusion_matrix(e['y_true'], e['y_pred'], labels=[0,1,2,3])
        for ii in range(4):
            for jj in range(4):
                cm_long.append({
                    'Population': pop,
                    'Model': m,
                    'True_Label': ii,
                    'Predicted_Label': jj,
                    'Count': cm[ii, jj]
                })
        ax = fig.add_subplot(gs[i, j])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, square=True)
        if i == 0:
            ax.set_title(m, fontsize=14)
        if j == 0:
            ax.set_ylabel(pop.replace('_', ' '), fontsize=14)
        ax.set_xlabel('Predicted' if i == len(POP_NAMES)-1 else '')

plt.suptitle('Confusion Matrices - All Populations and Models (All Risk Classes)',
             fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(op('all_population_model_confusion_matrices.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(op('all_population_model_confusion_matrices.png'), dpi=200, bbox_inches='tight')
plt.close()

df_cm = pd.DataFrame(cm_long)
df_cm.to_excel(op('all_confusion_matrices_data.xlsx'), index=False)

print(f"\n完成！所有结果已保存到: {os.path.abspath(OUT)}")
print("\n测试集 ACC:")
print(df_acc)
print("\n测试集 AUC:")
print(df_auc)
print("\n注意: SHAP图已输出所有风险等级（class_0, class_1, class_2, class_3）的结果")
