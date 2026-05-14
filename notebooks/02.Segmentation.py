# =============================================================================
# RideWise --- Customer Segmentation
# Purpose: Group all riders into behavioral clusters to tailor retention strategies
# =============================================================================


# =============================================================================
# SETUP & IMPORTS
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Define data paths
DATA_PROCESSED = os.path.join("..", "data", "processed")
MODEL_DIR      = os.path.join("..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

features = pd.read_csv(os.path.join(DATA_PROCESSED, "features.csv"))


# =============================================================================
# STEP 1 - Select Clustering Features
# =============================================================================
# Features used for segmentation:
#   1 - RFM features
#   2 - Trip behaviour
#   3 - Session engagement features
#   4 - Temporal features

print(features.head())

CLUSTER_COLS = [
    # RFM Score Features
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",

    # Trip behaviour
    "avg_fare", "avg_surge", "tip_rate", "peak_hour_rate", "weekend_ratio",

    # Session engagement
    "session_last_30d", "avg_time_on_app", "session_conversation_rate", "engagement_score",

    # Temporal features
    "account_age_days", "activity_trend_30d",
]

# K-Means clustering can't handle missing values — drop them
seg_df = features[["user_id", "churned"] + CLUSTER_COLS].dropna().copy()
print(f"Riders for segmentation: {len(seg_df)}")


# =============================================================================
# STEP 2 - Scaling
# =============================================================================
# StandardScaler converts each feature to mean=0, std=1 so all features
# contribute equally to distance calculations in K-Means.

scaler = StandardScaler()
X = scaler.fit_transform(seg_df[CLUSTER_COLS])


# =============================================================================
# STEP 3 - Elbow + Silhouette Test: Finding the Right Number of Clusters
# =============================================================================
# - Inertia  : sum of squared distances from each point to its cluster center (lower = better)
# - Silhouette: measures how well-separated clusters are (range -1 to 1, closer to 1 = better)

inertias     = []
silhouettes  = []
K_RANGE      = range(2, 9)

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=3000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), inertias, "bo-")
axes[0].set(title="Elbow Curve (Inertia)", xlabel="k", ylabel="Inertia")

axes[1].plot(list(K_RANGE), silhouettes, "go-")
axes[1].set(title="Silhouette Score", xlabel="k", ylabel="Score")

plt.tight_layout()
plt.show()


# =============================================================================
# STEP 4 - Fit Final Model with K=4
# =============================================================================
# Reasoning:
#   Elbow chart  : inertia drops sharply until k=4, then flattens (diminishing returns)
#   Silhouette   : k=4 has the best separation score among meaningful options

K_FINAL = 4
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
seg_df["cluster"] = kmeans.fit_predict(X)

final_sil = silhouette_score(X, seg_df["cluster"], sample_size=3000, random_state=42)
print(f"k={K_FINAL} | Inertia: {kmeans.inertia_:,.0f} | Silhouette: {final_sil:.2f}")
print(seg_df["cluster"].value_counts().sort_index())


# =============================================================================
# STEP 5 - Cluster Profiling
# =============================================================================
# Show the average behaviour of each cluster to understand what makes them unique.

profile_cols = [
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "avg_fare", "session_last_30d", "engagement_score", "activity_trend_30d"
]

profile = seg_df.groupby("cluster")[profile_cols].mean().round(2)
print(profile.T.to_string())

# Heatmap — normalise profile to 0-1 for colour scale, show raw values as annotations
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    profile_norm.T, annot=profile.T, fmt=".2f",
    cmap="YlOrRd", linewidths=0.5, ax=ax, cbar=False
)
ax.set_title("Cluster Profiles (raw values, normalised colour)", fontweight="bold")
ax.set_xlabel("Cluster")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6 - Assign Business Labels
# =============================================================================

rfm_mean = seg_df.groupby("cluster")[[
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"
]].mean().mean(axis=1)

rank = rfm_mean.rank(ascending=False).astype(int)

SEGMENT_LABELS = {}
names = ["Champions", "Loyal Riders", "At-Risk", "Dormant"]

for cluster_id, r in rank.items():
    SEGMENT_LABELS[cluster_id] = names[r - 1]

seg_df["segment"] = seg_df["cluster"].map(SEGMENT_LABELS)
print(seg_df["segment"].value_counts())


# =============================================================================
# STEP 6b - Churn Overlap: Do Segments Correlate with Churn?
# =============================================================================

churn_by_seg = seg_df.groupby("segment")["churned"].mean().mul(100).sort_values(ascending=False)
seg_count    = seg_df["segment"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Bar chart — churn rate per segment
bars = axes[0].bar(churn_by_seg.index, churn_by_seg.values, edgecolor="white")
for bar, v in zip(bars, churn_by_seg.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, v + 0.5,
        f"{v:.1f}", ha="center", fontsize=10
    )
axes[0].set(title="Churn Rate by Segment", ylabel="Churn %")

# Pie chart — segment size distribution
axes[1].pie(seg_count.values, labels=seg_count.index, autopct="%1.1f%%", startangle=140)
axes[1].set_title("Segment Distribution")

plt.suptitle("Customer Segments vs Churn", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Business Interpretation:
#   Champions  (~9.9% churn,  36.1% of riders) — Recent, frequent, high spenders. Reward loyalty.
#   At-Risk    (~10.6% churn, 38.3% of riders) — Previously active, now declining. Personal outreach + retention discounts.
#   Dormant    (~12.2% churn, 19.2% of riders) — Inactive for 30+ days. Low-cost automated "We miss you" campaign.


# =============================================================================
# STEP 7 - Save Assignments & Model Artifacts
# =============================================================================

# Save segment assignments
assignments = seg_df[["user_id", "cluster", "segment"]]
assignments.to_csv(os.path.join(DATA_PROCESSED, "segment_assignment.csv"), index=False)
print("Saved: segment_assignment.csv")

# Save model artifacts
joblib.dump(kmeans,       os.path.join(MODEL_DIR, "kmeans_segmentation.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler_segmentation.pkl"))
joblib.dump(CLUSTER_COLS, os.path.join(MODEL_DIR, "segmentation_features_cols.pkl"))

print("Saved all model artifacts.")