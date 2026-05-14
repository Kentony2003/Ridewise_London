# =============================================================================
# RideWise --- Churn Modelling & Evaluation
# Purpose: Train models to predict which rider will churn, evaluate performance
#          and select the best model.
#
# Models:
#   1. Logistic Regression  — Simple, fast, interpretable (Baseline)
#   2. Random Forest        — Robust, handles non-linear relationships
#   3. XGBoost              — Advanced, best performance, handles imbalanced data
# =============================================================================


# =============================================================================
# STEP 1 - Setup & Load Data
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.2)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier

# Define paths — resolved relative to this script's location so it works
# no matter which directory you run it from
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MODEL_DIR      = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

features   = pd.read_csv(os.path.join(DATA_PROCESSED, "features.csv"))
segment_df = pd.read_csv(os.path.join(DATA_PROCESSED, "segment_assignment.csv"))

print(features.head())


# =============================================================================
# STEP 1b - Drop rows with missing churn label
# =============================================================================

df = features.dropna(subset=["churned"]).copy()
churn_rate = df["churned"].mean()
print(f"Riders : {len(df):,}")
print(f"Churned: {df['churned'].sum():,} ({churn_rate * 100:.1f}%)")


# =============================================================================
# STEP 2 - Features & Train/Test Split
# =============================================================================

# Exclude identifiers and the target label
EXCLUDE      = ["user_id", "churned"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

print(f"\n{len(FEATURE_COLS)} features selected:")
print(FEATURE_COLS)

X = df[FEATURE_COLS].values
y = df["churned"].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# 80/20 train-test split, stratified to preserve churn rate in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale positive weight for XGBoost (handles class imbalance)
# Formula: neg / pos — tells XGBoost churned riders are X times rarer
neg, pos   = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos  = neg / pos

print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,} | Scale_pos_weight: {scale_pos:.1f}")


# =============================================================================
# STEP 2b - Define Models
# =============================================================================

# 1. Logistic Regression — Baseline
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=500,
    random_state=42
)

# 2. Random Forest — Ensemble of 200 decision trees
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 3. XGBoost — Sequential boosting with class imbalance handling
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# =============================================================================
# STEP 2e - Train All Models & Print Scores
# =============================================================================

MODELS = {
    "Logistic Regression": lr,
    "Random Forest":       rf,
    "XGBoost":             xgb
}

probas, preds = {}, {}

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    probas[name] = model.predict_proba(X_test)[:, 1]
    preds[name]  = model.predict(X_test)
    print(
        f"{name:25s}  AUC: {roc_auc_score(y_test, probas[name]):.4f}"
        f"  | F1: {f1_score(y_test, preds[name]):.4f}"
    )


# =============================================================================
# STEP 2f - ROC & Precision-Recall Curves
# =============================================================================

colors = ["blue", "orange", "green"]
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

for (name, proba), color in zip(probas.items(), colors):
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0], color=color)
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1], color=color)

axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right", fontsize=10)

axes[1].axhline(
    y=churn_rate, color="red", linestyle="--",
    label="Baseline (churn rate: {:.1f}%)".format(churn_rate * 100)
)
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right", fontsize=10)

plt.suptitle("Model Evaluation", fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 3 - Threshold Tuning (Best Model)
# =============================================================================

best_name  = max(probas, key=lambda n: roc_auc_score(y_test, probas[n]))
best_proba = probas[best_name]
print(f"\nBest model: {best_name} with AUC: {roc_auc_score(y_test, best_proba):.4f}")

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, best_proba)

# Calculate F1 at each threshold (exclude last element — no matching threshold)
f1_scores        = 2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1])
optimal_idx       = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1        = f1_scores[optimal_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_scores,              color="blue",   label="F1 Score",  linewidth=2)
ax.plot(thresholds, precision_vals[:-1],    color="orange", label="Precision", linestyle="--")
ax.plot(thresholds, recall_vals[:-1],       color="green",  label="Recall",    linestyle="--")
ax.axvline(
    optimal_threshold, color="red", linestyle="--",
    label=f"Optimal Threshold: {optimal_threshold:.2f} | F1: {optimal_f1:.2f}"
)
ax.set(title=f"Threshold vs F1 Score (Best Model: {best_name})", xlabel="Threshold", ylabel="Score")
ax.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 4 - Confusion Matrix
# =============================================================================

best_pred_opt = (best_proba >= optimal_threshold).astype(int)
best_pred_def = (best_proba >= 0.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pred, label in zip(
    axes,
    [best_pred_def, best_pred_opt],
    ["Default Threshold (0.5)", f"Optimal Threshold ({optimal_threshold:.2f})"]
):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Active", "Churned"],
        yticklabels=["Active", "Churned"]
    )
    ax.set(title=label, xlabel="Predicted", ylabel="Actual")

plt.suptitle(f"Confusion Matrix — {best_name}", fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 5 - Segment-Level Performance
# =============================================================================

test_results = pd.DataFrame({
    "user_id": df["user_id"].values,
    "y_true":  y,
    "y_proba": MODELS[best_name].predict_proba(df[FEATURE_COLS].values)[:, 1],
}).merge(segment_df[["user_id", "segment"]], on="user_id", how="left")

test_results["y_pred"] = (test_results["y_proba"] >= optimal_threshold).astype(int)

seg_perf = (
    test_results.groupby("segment")
    .apply(lambda g: pd.Series({
        "n_riders":             len(g),
        "actual_churn_pct":     g["y_true"].mean() * 100,
        "predicted_churn_pct":  g["y_pred"].mean() * 100,
        "avg_churn_proba":      g["y_proba"].mean(),
        "auc": roc_auc_score(g["y_true"], g["y_proba"]) if g["y_true"].nunique() > 1 else np.nan
    })).round(2)
)

print(seg_perf.to_string())

# Visualise segment performance
seg_order = seg_perf["actual_churn_pct"].sort_values(ascending=False).index
fig, ax   = plt.subplots(figsize=(10, 6))
x         = np.arange(len(seg_perf))
width     = 0.35

ax.bar(x - width/2, seg_perf.loc[seg_order, "actual_churn_pct"],    width=width, label="Actual Churn %",    edgecolor="white")
ax.bar(x + width/2, seg_perf.loc[seg_order, "predicted_churn_pct"], width=width, label="Predicted Churn %", edgecolor="white")

for i, v in enumerate(seg_perf.loc[seg_order, "actual_churn_pct"]):
    ax.text(i - width/2, v + 0.2, f"{v:.1f}%", ha="center")
for i, v in enumerate(seg_perf.loc[seg_order, "predicted_churn_pct"]):
    ax.text(i + width/2, v + 0.2, f"{v:.1f}%", ha="center")

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(seg_order, rotation=45)
ax.set(title="Actual vs Predicted Churn Rate by Segment", ylabel="Churn Rate (%)")
ax.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6 - Feature Importance
# =============================================================================

best_model = MODELS[best_name]

if hasattr(best_model, "feature_importances_"):
    importance = (
        pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax, color="skyblue", edgecolor="black")
    ax.set(title=f"Top 15 Feature Importances — {best_name}", xlabel="Importance Score")
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 7 - Results Summary & Save Artifacts
# =============================================================================

rows = []
for name, model in MODELS.items():
    rows.append({
        "model":         name,
        "roc_auc":       round(roc_auc_score(y_test, probas[name]), 4),
        "f1_default":    round(f1_score(y_test, preds[name]), 4),
        "f1_optimal":    round(f1_score(y_test, (probas[name] >= optimal_threshold).astype(int)), 4),
        "precision_def": round(precision_score(y_test, preds[name]), 4),
        "recall_def":    round(recall_score(y_test, preds[name]), 4),
    })

results_df = pd.DataFrame(rows)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Save artifacts
joblib.dump(best_model,        os.path.join(MODEL_DIR, "churn_model.pkl"))
joblib.dump(FEATURE_COLS,      os.path.join(MODEL_DIR, "feature_cols.pkl"))
joblib.dump(optimal_threshold, os.path.join(MODEL_DIR, "optimal_threshold.pkl"))
print("\nModel and artifacts saved successfully!")