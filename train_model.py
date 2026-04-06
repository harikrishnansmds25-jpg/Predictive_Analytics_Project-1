"""
Mental Health Status Classification - Model Training Script
Run this once to train and save the model as model.pkl
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATASET ─────────────────────────────────────────────────────────
# Dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
# Download and place as 'mental_health.csv' in the same folder
# Expected columns: 'statement' (text) and 'status' (label)

def load_data(filepath='mental_health.csv'):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nClass distribution:\n{df['status'].value_counts()}")
    return df

# ── 2. PREPROCESS TEXT ──────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)             # remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # clean whitespace
    return text

def preprocess(df):
    print("\nPreprocessing text...")
    df = df.dropna(subset=['statement', 'status'])
    df['clean_text'] = df['statement'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 5]

    # Normalize labels
    label_map = {
        'Depression': 'Depression',
        'Anxiety': 'Anxiety',
        'PTSD': 'PTSD',
        'Normal': 'Normal',
        'Suicidal': 'Depression',       # merge suicidal with depression
        'Stress': 'Anxiety',            # merge stress with anxiety
        'Bipolar': 'Depression',
        'Personality disorder': 'PTSD'
    }
    df['label'] = df['status'].map(label_map)
    df = df.dropna(subset=['label'])
    print(f"After preprocessing: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df

# ── 3. TRAIN MODELS ─────────────────────────────────────────────────────────
def train_and_evaluate(df):
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
        'Linear SVM':          LinearSVC(max_iter=1000, C=1.0, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    results = {}
    best_acc = 0
    best_pipeline = None
    best_name = ''

    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                sublinear_tf=True
            )),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if acc > best_acc:
            best_acc = acc
            best_pipeline = pipeline
            best_name = name

    print(f"\n{'='*60}")
    print(f"Best Model: {best_name} with accuracy {best_acc:.4f}")
    print(f"{'='*60}")

    return best_pipeline, best_name, results, X_test, y_test

# ── 4. SAVE MODEL ───────────────────────────────────────────────────────────
def save_model(pipeline, filepath='model.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {filepath}")

# ── 5. MAIN ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists('mental_health.csv'):
        print("ERROR: 'mental_health.csv' not found!")
        print("Download from: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health")
        print("Place it in the same folder as this script and run again.")
    else:
        df = load_data()
        df = preprocess(df)
        pipeline, best_name, results, X_test, y_test = train_and_evaluate(df)
        save_model(pipeline)
        print("\nDone! You can now run: streamlit run app.py")
