"""
03_train_bert.py  —  Fine-tune BERT for Mental Health Classification
=====================================================================
Dataset : data_to_be_cleansed.csv
Classes : 0=Stress | 1=Depression | 2=Bipolar | 3=Pers.Disorder | 4=Anxiety
Saves   : bert_model/            (HuggingFace save_pretrained)
          bert_predictions.pkl   (y_pred, y_true, y_prob on test set)
          bert_train_results.png
Run     : python 03_train_bert.py
Note    : GPU strongly recommended (15-30 min on GPU, hours on CPU).
          For a quick CPU smoke-test set EPOCHS=1 and MAX_LEN=64.
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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (BertTokenizer, BertForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data_to_be_cleansed.csv"
LABEL_MAP    = {0: 'Stress', 1: 'Depression', 2: 'Bipolar', 3: 'Pers. Disorder', 4: 'Anxiety'}
ORDER        = ['Depression', 'Anxiety', 'Stress', 'Bipolar', 'Pers. Disorder']
BERT_MODEL   = 'bert-base-uncased'
MAX_LEN      = 128      # reduce to 64 for CPU smoke-test
BATCH_SIZE   = 16
EPOCHS       = 3        # increase to 4-5 for best accuracy
LR           = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE    = 0.2
RANDOM_STATE = 42
NUM_CLASSES  = 5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device : {DEVICE}")
if DEVICE.type == 'cpu':
    print("WARNING: Running on CPU will be very slow. Consider reducing MAX_LEN=64, EPOCHS=1.")

# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df = pd.read_csv(DATA_PATH)
df['label']     = df['target'].map(LABEL_MAP)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df              = df.dropna(subset=['full_text'])
df['clean']     = df['full_text'].apply(clean_text)

X = df['clean'].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")

# ── 2. Dataset Class ──────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = MentalHealthDataset(X_train, y_train, tokenizer, MAX_LEN)
test_ds  = MentalHealthDataset(X_test,  y_test,  tokenizer, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 3. Model Setup ────────────────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_CLASSES)
model.to(DEVICE)

optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_dl) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# ── 4. Training Loop ──────────────────────────────────────────────────────────
train_losses, train_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for step, batch in enumerate(train_dl):
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels    = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

        if (step + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_dl)} "
                  f"| Loss {running_loss/(step+1):.4f} | Acc {correct/total:.3f}")

    train_losses.append(running_loss / len(train_dl))
    train_accs.append(correct / total)
    print(f"\nEpoch {epoch} done — Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.3f}\n")

# ── 5. Test Evaluation ────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in test_dl:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels    = batch['labels']

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.cpu()
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = logits.argmax(dim=1).numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy : {test_acc:.3f}")
print(classification_report(all_labels, all_preds, target_names=ORDER))

# ── 6. Save Model & Predictions ───────────────────────────────────────────────
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')
print("✓ Saved  →  bert_model/")

with open('bert_predictions.pkl', 'wb') as f:
    pickle.dump({
        'y_pred':        np.array(all_preds),
        'y_true':        np.array(all_labels),
        'y_prob':        np.array(all_probs),
        'train_losses':  train_losses,
        'train_accs':    train_accs,
    }, f)
print("✓ Saved  →  bert_predictions.pkl")

# ── 7. Training Curve Plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('BERT Fine-Tuning — Training Results', fontsize=15, fontweight='bold')

epochs_x = list(range(1, EPOCHS + 1))

# Loss curve
ax = axes[0]
ax.plot(epochs_x, train_losses, 'o-', color='#c16b8a', linewidth=2.5, markersize=8)
for x, y_val in zip(epochs_x, train_losses):
    ax.text(x, y_val + 0.015, f'{y_val:.3f}', ha='center', fontsize=10,
            color='#c16b8a', fontweight='bold')
ax.fill_between(epochs_x, train_losses, alpha=0.12, color='#c16b8a')
ax.set_title('Training Loss per Epoch', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.set_xticks(epochs_x); ax.set_ylim(0, max(train_losses) * 1.3)
sns.despine(ax=ax)

# Accuracy curve
ax = axes[1]
ax.plot(epochs_x, train_accs, 'o-', color='#5b7fc1', linewidth=2.5, markersize=8,
        label='Train Acc')
ax.axhline(test_acc, linestyle='--', color='#e07b54', linewidth=2,
           label=f'Test Acc = {test_acc:.3f}')
for x, y_val in zip(epochs_x, train_accs):
    ax.text(x, y_val + 0.008, f'{y_val:.3f}', ha='center', fontsize=10,
            color='#5b7fc1', fontweight='bold')
ax.fill_between(epochs_x, train_accs, alpha=0.10, color='#5b7fc1')
ax.set_title('Training Accuracy per Epoch', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_xticks(epochs_x); ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=10); sns.despine(ax=ax)

# Test confusion matrix
ax = axes[2]
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=ORDER).plot(
    ax=ax, colorbar=False, cmap='Oranges')
ax.set_title(f'Test Confusion Matrix\n(Acc = {test_acc:.3f})', fontweight='bold')
ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=8)
ax.set_yticklabels(ORDER, fontsize=8)

fig.tight_layout()
fig.savefig('bert_train_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved  →  bert_train_results.png")
