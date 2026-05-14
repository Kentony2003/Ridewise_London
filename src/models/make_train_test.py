import os, sys
sys.path.insert(0, ".")
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = r"C:\Users\THINKPAD\OneDrive\Desktop\DATA FILES\Ridewise_London PY"
df = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"))
print(f"Loaded {len(df):,} rows")
print(f"Churn distribution:\n{df['churned'].value_counts()}\n")

train, test = train_test_split(df, test_size=0.20, random_state=42, stratify=df["churned"])
print(f"Train : {len(train):,} rows  |  churn rate: {train['churned'].mean():.2%}")
print(f"Test  : {len(test):,} rows  |  churn rate: {test['churned'].mean():.2%}")

train.to_csv(os.path.join(BASE, "data", "processed", "train.csv"), index=False)
test.to_csv(os.path.join(BASE,  "data", "processed", "test.csv"),  index=False)
print("Saved train.csv and test.csv")
