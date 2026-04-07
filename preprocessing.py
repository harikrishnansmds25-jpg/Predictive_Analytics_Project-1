"""
Mental Health Status Classification
Data Preprocessing & Cleaning Pipeline
Author: Umapravathy C S
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD RAW DATA ─────────────────────────────────────────────────────────
def load_data(filepath='mental_health.csv'):
    print("Loading raw data...")
    df = pd.read_csv(filepath)
    print(f"Raw dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    return df

# ── 2. CLEAN TEXT ────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                    # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)            # remove URLs
    text = re.sub(r'@\w+', '', text)                      # remove mentions
    text = re.sub(r'#\w+', '', text)                      # remove hashtags
    text = re.sub(r'\d+', '', text)                       # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)                  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()              # remove extra spaces
    return text

# ── 3. NORMALIZE LABELS ──────────────────────────────────────────────────────
def normalize_labels(df):
    print("\nNormalizing labels...")
    label_map = {
        'Depression': 'Depression',
        'Anxiety': 'Anxiety',
        'PTSD': 'PTSD',
        'Normal': 'Normal',
        'Suicidal': 'Depression',
        'Stress': 'Anxiety',
        'Bipolar': 'Depression',
        'Personality disorder': 'PTSD'
    }
    df['label'] = df['status'].map(label_map)
    print(f"Label distribution after normalization:\n{df['label'].value_counts()}")
    return df

# ── 4. HANDLE MISSING VALUES ─────────────────────────────────────────────────
def handle_missing(df):
    print("\nHandling missing values...")
    before = len(df)
    df = df.dropna(subset=['statement', 'status'])
    after = len(df)
    print(f"Removed {before - after} rows with missing values")
    return df

# ── 5. REMOVE DUPLICATES ─────────────────────────────────────────────────────
def remove_duplicates(df):
    print("\nRemoving duplicates...")
    before = len(df)
    df = df.drop_duplicates(subset=['statement'])
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df

# ── 6. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineering(df):
    print("\nEngineering features...")
    df['clean_text'] = df['statement'].apply(clean_text)
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['clean_text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df['exclamation_count'] = df['statement'].astype(str).apply(lambda x: x.count('!'))
    df['question_count'] = df['statement'].astype(str).apply(lambda x: x.count('?'))

    # Remove very short texts
    before = len(df)
    df = df[df['word_count'] >= 3]
    print(f"Removed {before - len(df)} rows with less than 3 words")
    print(f"\nNew features added: text_length, word_count, avg_word_length, exclamation_count, question_count")
    return df

# ── 7. SAVE PROCESSED DATA ───────────────────────────────────────────────────
def save_processed(df, filepath='processed_data.csv'):
    df.to_csv(filepath, index=False)
    print(f"\nProcessed data saved to {filepath}")
    print(f"Final dataset shape: {df.shape}")

# ── 8. SUMMARY REPORT ────────────────────────────────────────────────────────
def summary_report(df):
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY REPORT")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    print(f"\nText length stats:\n{df['text_length'].describe()}")
    print(f"\nWord count stats:\n{df['word_count'].describe()}")
    print("="*50)

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists('mental_health.csv'):
        print("ERROR: 'mental_health.csv' not found!")
        print("Download from: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health")
    else:
        df = load_data()
        df = handle_missing(df)
        df = remove_duplicates(df)
        df = normalize_labels(df)
        df = feature_engineering(df)
        summary_report(df)
        save_processed(df)
        print("\nPreprocessing complete!")
