"""
04_test_models.py  —  Evaluate Both Models on Held-Out Test Set
================================================================
Loads  : tfidf_svm_model.pkl   (from 02_train_tfidf_svm.py)
         bert_predictions.pkl   (from 03_train_bert.py)
         X_test.pkl  y_test.pkl (from 02_train_tfidf_svm.py)
Saves  : tfidf_svm_test_results.png
         bert_test_results.png
Run    : python 04_test_models.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_MAP = {0: 'Stress', 1: 'Depression', 2: 'Bipolar', 3: 'Pers. Disorder', 4: 'Anxiety'}
ORDER     = ['Depression', 'Anxiety', 'Stress', 'Bipolar', 'Pers. Disorder']
PALETTE   = {
    'Depression': '#5b7fc1', 'Anxiety': '#e07b54', 'Stress': '#f5c842',
    'Bipolar': '#6bb58c', 'Pers. Disorder': '#c16b8a',
}

# ── Helper: softmax for SVM decision scores ───────────────────────────────────
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ── Helper: draw 3-panel evaluation figure ───────────────────────────────────
def evaluation_figure(y_true, y_pred, y_prob, title_prefix, out_file, cmap='Blues'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{title_prefix} — Test Set Evaluation',
                 fontsize=15, fontweight='bold')

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=ORDER).plot(
        ax=ax, colorbar=False, cmap=cmap)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ORDER, fontsize=9)

    # Per-class Precision / Recall / F1
    ax = axes[1]
    labels_sorted = sorted(set(y_true))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted)
    x_pos = np.arange(len(ORDER)); w = 0.25
    ax.bar(x_pos - w, prec, w, label='Precision', color='#5b7fc1', edgecolor='white')
    ax.bar(x_pos,     rec,  w, label='Recall',    color='#e07b54', edgecolor='white')
    ax.bar(x_pos + w, f1,   w, label='F1-Score',  color='#6bb58c', edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ORDER, rotation=20, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics', fontweight='bold')
    ax.legend(fontsize=9); sns.despine(ax=ax)

    # One-vs-Rest ROC curves
    ax = axes[2]
    for i, lbl in enumerate(ORDER):
        class_idx   = [k for k, v in LABEL_MAP.items() if v == lbl][0]
        binary_true = (np.array(y_true) == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, class_idx])
        auc         = roc_auc_score(binary_true, y_prob[:, class_idx])
        ax.plot(fpr, tpr, label=f'{lbl} (AUC={auc:.2f})',
                color=PALETTE[lbl], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('One-vs-Rest ROC Curves', fontweight='bold')
    ax.legend(fontsize=8); sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved  →  {out_file}")

# ── 1. Load Test Data ─────────────────────────────────────────────────────────
with open('X_test.pkl', 'rb') as f: X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f: y_test = pickle.load(f)

print(f"Test samples : {len(y_test)}")
print(pd.Series(y_test).map(LABEL_MAP).value_counts().to_string())

# ── 2. TF-IDF + SVM Evaluation ───────────────────────────────────────────────
print("\n" + "="*55)
print("  TF-IDF + SVM — Test Evaluation")
print("="*55)

with open('tfidf_svm_model.pkl', 'rb') as f:
    svm_pipe = pickle.load(f)

y_pred_svm = svm_pipe.predict(X_test)
y_prob_svm = softmax(svm_pipe.decision_function(X_test))

print(f"\nTest Accuracy : {(y_pred_svm == y_test.values).mean():.3f}")
print(classification_report(y_test, y_pred_svm, target_names=ORDER))

evaluation_figure(y_test.values, y_pred_svm, y_prob_svm,
                  'TF-IDF + SVM', 'tfidf_svm_test_results.png', cmap='Blues')

# ── 3. BERT Evaluation ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  BERT — Test Evaluation")
print("="*55)

with open('bert_predictions.pkl', 'rb') as f:
    bert_data = pickle.load(f)

y_pred_bert = bert_data['y_pred']
y_true_bert = bert_data['y_true']
y_prob_bert = bert_data['y_prob']

print(f"\nTest Accuracy : {(y_pred_bert == y_true_bert).mean():.3f}")
print(classification_report(y_true_bert, y_pred_bert, target_names=ORDER))

evaluation_figure(y_true_bert, y_pred_bert, y_prob_bert,
                  'BERT', 'bert_test_results.png', cmap='Oranges')
