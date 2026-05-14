"""
Run this once to create train.csv and test.csv from features.csv.

Usage:
    $env:PYTHONPATH = "."; & "C:/Program Files/Python314/python.exe" "src/models/make_train_test.py"
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\THINKPAD\OneDrive\Desktop\DATA FILES\Ridewise_London PY"
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "features.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Churn distribution:\n{df['churned'].value_counts()}\n")

# ── split 80 / 20, stratified on churn ────────────────────────────────────────
train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["churned"]          # keeps churn ratio equal in both splits
)

print(f"Train : {len(train):,} rows  |  churn rate: {train['churned'].mean():.2%}")
print(f"Test  : {len(test):,} rows  |  churn rate: {test['churned'].mean():.2%}")

# ── save ───────────────────────────────────────────────────────────────────────
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)

print(f"\nSaved train.csv → {train_path}")
print(f"Saved test.csv  → {test_path}")