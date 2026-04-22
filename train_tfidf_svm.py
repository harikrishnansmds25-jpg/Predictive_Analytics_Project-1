"""
02_train_tfidf_svm.py  —  Train TF-IDF + LinearSVC
====================================================
Dataset : data_to_be_cleansed.csv
Saves   : tfidf_svm_model.pkl
          X_test.pkl  y_test.pkl
          tfidf_svm_train_results.png
Run     : python 02_train_tfidf_svm.py
"""

import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data_to_be_cleansed.csv"
LABEL_MAP = {0: 'Stress', 1: 'Depression', 2: 'Bipolar', 3: 'Pers. Disorder', 4: 'Anxiety'}
ORDER     = ['Depression', 'Anxiety', 'Stress', 'Bipolar', 'Pers. Disorder']
PALETTE   = {
    'Depression': '#5b7fc1', 'Anxiety': '#e07b54', 'Stress': '#f5c842',
    'Bipolar': '#6bb58c', 'Pers. Disorder': '#c16b8a',
}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters
    return re.sub(r'\s+', ' ', text).strip()

df = pd.read_csv(DATA_PATH)
df['label']     = df['target'].map(LABEL_MAP)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df              = df.dropna(subset=['full_text'])
df['clean']     = df['full_text'].apply(clean_text)

X = df['clean']
y = df['target']

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")
print("Train class distribution:")
print(y_train.map(LABEL_MAP).value_counts().to_string())

# ── 3. TF-IDF + SVM Pipeline ─────────────────────────────────────────────────
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15_000,   # vocabulary cap
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log-TF scaling (reduces high-freq dominance)
        min_df=3,              # ignore very rare terms
        max_df=0.90,           # ignore near-universal terms
        stop_words='english',  # remove functional words
    )),
    ('svm', LinearSVC(C=0.8, max_iter=2000)),
])

# ── 4. 5-Fold Cross-Validation on training set ───────────────────────────────
print("\nRunning 5-fold cross-validation …")
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')
print(f"  Fold scores : {cv_scores.round(3)}")
print(f"  Mean ± Std  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 5. Fit on Full Training Set ───────────────────────────────────────────────
pipe.fit(X_train, y_train)
train_acc = pipe.score(X_train, y_train)
print(f"\nTrain accuracy : {train_acc:.3f}")
print("\nClassification report (train set):")
print(classification_report(y_train, pipe.predict(X_train), target_names=ORDER))

# ── 6. Save Artefacts ─────────────────────────────────────────────────────────
with open('tfidf_svm_model.pkl', 'wb') as f:
    pickle.dump(pipe, f)
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
print("\n✓ Saved  →  tfidf_svm_model.pkl | X_test.pkl | y_test.pkl")

# ── 7. Training Diagnostics Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('TF-IDF + SVM — Training Diagnostics', fontsize=15, fontweight='bold')

# 7a — Cross-val bar chart
ax = axes[0]
bars = ax.bar(range(1, 6), cv_scores, color='#a0a0c8', edgecolor='white', width=0.6)
ax.axhline(cv_scores.mean(), color='#c16b8a', linestyle='--', linewidth=2,
           label=f'Mean = {cv_scores.mean():.3f}')
ax.fill_between(range(1, 6),
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.18, color='#c16b8a')
for bar, s in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{s:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(1, 6)); ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
ax.set_ylim(0.6, 1.0)
ax.set_title('5-Fold CV Accuracy (Train Set)', fontweight='bold')
ax.legend(); sns.despine(ax=ax)

# 7b — Top-20 highest IDF (most distinctive) terms
tfidf    = pipe.named_steps['tfidf']
feats    = tfidf.get_feature_names_out()
idf_vals = tfidf.idf_
top20    = np.argsort(idf_vals)[-20:][::-1]
ax = axes[1]
ax.barh(feats[top20][::-1], idf_vals[top20][::-1], color='#5b7fc1', edgecolor='white')
ax.set_title('Top-20 High-IDF (Rare/Distinctive) Terms', fontweight='bold')
ax.set_xlabel('IDF Score')
sns.despine(ax=ax)

# 7c — Mean absolute SVM weight per class
svm_coef  = pipe.named_steps['svm'].coef_
mean_mags = np.abs(svm_coef).mean(axis=1)
ax = axes[2]
ax.bar(ORDER, mean_mags, color=[PALETTE[l] for l in ORDER], edgecolor='white')
ax.set_title('Mean |SVM Weight| per Class', fontweight='bold')
ax.set_ylabel('Mean absolute coefficient')
ax.set_xticklabels(ORDER, rotation=18, fontsize=9)
sns.despine(ax=ax)

fig.tight_layout()
fig.savefig('tfidf_svm_train_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved  →  tfidf_svm_train_results.png")
