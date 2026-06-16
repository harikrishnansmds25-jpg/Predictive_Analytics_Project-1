"""
05_compare_models.py  —  Side-by-Side Model Comparison
=======================================================
Loads  : tfidf_svm_model.pkl | bert_predictions.pkl | X_test.pkl | y_test.pkl
Saves  : model_comparison.png
Run    : python 05_compare_models.py
"""

import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_MAP = {0: 'Stress', 1: 'Depression', 2: 'Bipolar', 3: 'Pers. Disorder', 4: 'Anxiety'}
ORDER     = ['Depression', 'Anxiety', 'Stress', 'Bipolar', 'Pers. Disorder']

# ── Load ──────────────────────────────────────────────────────────────────────
with open('X_test.pkl',          'rb') as f: X_test    = pickle.load(f)
with open('y_test.pkl',          'rb') as f: y_test    = pickle.load(f)
with open('tfidf_svm_model.pkl', 'rb') as f: svm_pipe  = pickle.load(f)
with open('bert_predictions.pkl','rb') as f: bert_data = pickle.load(f)

# SVM inference
t0           = time.perf_counter()
y_pred_svm   = svm_pipe.predict(X_test)
svm_infer_ms = (time.perf_counter() - t0) * 1000

# BERT
y_pred_bert = bert_data['y_pred']
y_true_bert = bert_data['y_true']

# ── Metrics ───────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
    return {
        'Accuracy':        accuracy_score(y_true, y_pred),
        'Macro F1':        f1_score(y_true, y_pred, average='macro'),
        'Macro Precision': precision_score(y_true, y_pred, average='macro'),
        'Macro Recall':    recall_score(y_true, y_pred, average='macro'),
        'Weighted F1':     f1_score(y_true, y_pred, average='weighted'),
    }

svm_metrics  = get_metrics(y_test.values,  y_pred_svm)
bert_metrics = get_metrics(y_true_bert, y_pred_bert)
svm_f1_cls   = f1_score(y_test.values,  y_pred_svm,  average=None)
bert_f1_cls  = f1_score(y_true_bert, y_pred_bert, average=None)

print("=" * 60)
print("         MODEL COMPARISON SUMMARY")
print("=" * 60)
df_m = pd.DataFrame({'TF-IDF + SVM': svm_metrics, 'BERT': bert_metrics})
print(df_m.round(4).to_string())
print(f"\nSVM inference time  : {svm_infer_ms:.1f} ms (test set)")
print("BERT inference time : ~30–120x slower")

# ── 4-panel comparison figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('TF-IDF + SVM  vs  BERT — Full Comparison',
             fontsize=17, fontweight='bold', y=1.01)

w = 0.35

# Panel A — Overall metrics
ax1 = fig.add_subplot(gs[0, 0])
metric_names = list(svm_metrics.keys())
x = np.arange(len(metric_names))
ax1.bar(x - w/2, [svm_metrics[m]  for m in metric_names], w,
        label='TF-IDF + SVM', color='#5b7fc1', edgecolor='white')
ax1.bar(x + w/2, [bert_metrics[m] for m in metric_names], w,
        label='BERT',         color='#e07b54', edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in metric_names], fontsize=8)
ax1.set_ylim(0, 1.15); ax1.set_ylabel('Score')
ax1.set_title('A  |  Overall Metrics', fontweight='bold', loc='left')
ax1.legend(fontsize=9); sns.despine(ax=ax1)
for i, (sv, bv) in enumerate(zip([svm_metrics[m] for m in metric_names],
                                  [bert_metrics[m] for m in metric_names])):
    ax1.text(i - w/2, sv + 0.01, f'{sv:.3f}', ha='center', fontsize=7, color='#5b7fc1')
    ax1.text(i + w/2, bv + 0.01, f'{bv:.3f}', ha='center', fontsize=7, color='#e07b54')

# Panel B — Per-class F1
ax2 = fig.add_subplot(gs[0, 1])
x2 = np.arange(len(ORDER))
ax2.bar(x2 - w/2, svm_f1_cls,  w, label='TF-IDF + SVM', color='#5b7fc1', edgecolor='white')
ax2.bar(x2 + w/2, bert_f1_cls, w, label='BERT',         color='#e07b54', edgecolor='white')
ax2.set_xticks(x2); ax2.set_xticklabels(ORDER, rotation=20, fontsize=9)
ax2.set_ylim(0, 1.15); ax2.set_ylabel('F1-Score')
ax2.set_title('B  |  Per-Class F1 Score', fontweight='bold', loc='left')
ax2.legend(fontsize=9); sns.despine(ax=ax2)

# Panel C — F1 delta (BERT gain)
ax3 = fig.add_subplot(gs[0, 2])
delta  = bert_f1_cls - svm_f1_cls
colors = ['#6bb58c' if d >= 0 else '#c16b8a' for d in delta]
ax3.bar(ORDER, delta, color=colors, edgecolor='white')
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title('C  |  BERT − SVM  F1 Delta', fontweight='bold', loc='left')
ax3.set_ylabel('F1 Gain')
ax3.set_xticklabels(ORDER, rotation=20, fontsize=9)
for i, d in enumerate(delta):
    ax3.text(i, d + (0.003 if d >= 0 else -0.010),
             f'{d:+.3f}', ha='center', fontsize=9,
             color='#2d7a4f' if d >= 0 else '#8b1a1a')
sns.despine(ax=ax3)

# Panel D — SVM confusion matrix
ax4 = fig.add_subplot(gs[1, 0])
ConfusionMatrixDisplay(confusion_matrix(y_test.values, y_pred_svm),
                       display_labels=ORDER).plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title('D  |  Confusion — TF-IDF + SVM', fontweight='bold', loc='left')
ax4.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=8)
ax4.set_yticklabels(ORDER, fontsize=8)

# Panel E — BERT confusion matrix
ax5 = fig.add_subplot(gs[1, 1])
ConfusionMatrixDisplay(confusion_matrix(y_true_bert, y_pred_bert),
                       display_labels=ORDER).plot(ax=ax5, colorbar=False, cmap='Oranges')
ax5.set_title('E  |  Confusion — BERT', fontweight='bold', loc='left')
ax5.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=8)
ax5.set_yticklabels(ORDER, fontsize=8)

# Panel F — Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Metric',         'TF-IDF + SVM',                         'BERT'],
    ['Accuracy',       f'{svm_metrics["Accuracy"]:.3f}',        f'{bert_metrics["Accuracy"]:.3f}'],
    ['Macro F1',       f'{svm_metrics["Macro F1"]:.3f}',        f'{bert_metrics["Macro F1"]:.3f}'],
    ['Macro Prec.',    f'{svm_metrics["Macro Precision"]:.3f}', f'{bert_metrics["Macro Precision"]:.3f}'],
    ['Macro Recall',   f'{svm_metrics["Macro Recall"]:.3f}',    f'{bert_metrics["Macro Recall"]:.3f}'],
    ['Train Time',     '< 30 sec (CPU)',                        '15–30 min (GPU)'],
    ['Interpretable',  '✓  Yes',                                '✗  No'],
    ['GPU Required',   '✗  No',                                 '✓  Yes (practical)'],
    ['Parameters',     '~15 K features',                        '110 M weights'],
]
tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for j in range(3):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(table_data)):
    tbl[(i, 1)].set_facecolor('#dce8f7')
    tbl[(i, 2)].set_facecolor('#fde9d9')
ax6.set_title('F  |  Model Trade-off Summary', fontweight='bold', loc='left', pad=10)
fig.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved  →  model_comparison.png")
