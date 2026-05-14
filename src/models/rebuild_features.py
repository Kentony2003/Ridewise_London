"""
rebuild_features.py
-------------------
Rebuilds features.csv with a properly defined churn label.

Churn definition:
    A rider is churned if they have made NO trips in the last 60 days
    relative to the latest date in the dataset.

Run:
    $env:PYTHONPATH = "."; & "C:/Program Files/Python314/python.exe" "src/data/rebuild_features.py"
"""

import os
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE  = r"C:\Users\THINKPAD\OneDrive\Desktop\DATA FILES\Ridewise_London PY"
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")

# ── load raw data ──────────────────────────────────────────────────────────────
print("Loading raw data...")
riders = pd.read_csv(os.path.join(RAW, "riders.csv"))
trips  = pd.read_csv(os.path.join(RAW, "trips.csv"))

# force datetime parsing regardless of format
riders["signup_date"] = pd.to_datetime(riders["signup_date"], infer_datetime_format=True)
trips["pickup_time"]  = pd.to_datetime(trips["pickup_time"],  infer_datetime_format=True)
trips["dropoff_time"] = pd.to_datetime(trips["dropoff_time"], infer_datetime_format=True)

print(f"  Riders : {len(riders):,}")
print(f"  Trips  : {len(trips):,}")
print(f"  Trip date range: {trips['pickup_time'].min().date()} -> {trips['pickup_time'].max().date()}")

# ── reference date ─────────────────────────────────────────────────────────────
snapshot = trips["pickup_time"].max()
print(f"\nSnapshot date: {snapshot.date()}")

# ── trip-level features per rider ─────────────────────────────────────────────
print("\nComputing trip features...")

trips["trip_duration_min"] = (
    trips["dropoff_time"] - trips["pickup_time"]
).dt.total_seconds() / 60

trips["days_ago"] = (snapshot - trips["pickup_time"]).dt.days

# helper: fare within last N days
def fare_last_n(group, n):
    return group.loc[trips.loc[group.index, "days_ago"] <= n, "fare"].sum()

agg = trips.groupby("user_id").agg(
    last_trip_days_ago  = ("days_ago",          "min"),
    trips_last_7d       = ("days_ago",          lambda x: (x <= 7).sum()),
    trips_last_30d      = ("days_ago",          lambda x: (x <= 30).sum()),
    trips_last_60d      = ("days_ago",          lambda x: (x <= 60).sum()),
    trips_last_90d      = ("days_ago",          lambda x: (x <= 90).sum()),
    trips_lifetime      = ("trip_id",           "count"),
    monetary_total      = ("fare",              "sum"),
    monetary_avg        = ("fare",              "mean"),
    avg_trip_duration   = ("trip_duration_min", "mean"),
    avg_fare            = ("fare",              "mean"),
    avg_surge           = ("surge_multiplier",  "mean"),
    tip_rate            = ("tip",               lambda x: (x > 0).mean()),
).reset_index()

# monetary_last_30d computed separately to avoid index alignment issues
m30 = trips[trips["days_ago"] <= 30].groupby("user_id")["fare"].sum().reset_index()
m30.columns = ["user_id", "monetary_last_30d"]
agg = agg.merge(m30, on="user_id", how="left")
agg["monetary_last_30d"] = agg["monetary_last_30d"].fillna(0)

# ── churn label: no trips in last 60 days ─────────────────────────────────────
agg["churned"] = (agg["last_trip_days_ago"] > 60).astype(int)
agg.rename(columns={"last_trip_days_ago": "days_since_last_trip"}, inplace=True)

print(f"\nChurn label distribution:")
vc = agg["churned"].value_counts()
print(vc)
print(f"Churn rate: {agg['churned'].mean():.2%}")

# ── merge with rider-level features ───────────────────────────────────────────
print("\nMerging with rider features...")
riders["account_age_days"] = (snapshot - riders["signup_date"]).dt.days
riders["was_referred"]     = riders["referred_by"].notna().astype(int)

loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
riders["loyalty_encoded"] = riders["loyalty_status"].map(loyalty_map).fillna(0).astype(int)

city_map = {c: i for i, c in enumerate(riders["city"].unique())}
riders["city_encoded"] = riders["city"].map(city_map)

rider_feats = riders[[
    "user_id", "account_age_days", "was_referred",
    "avg_rating_given", "loyalty_encoded", "city_encoded"
]]

df = agg.merge(rider_feats, on="user_id", how="left")

# ── validation ────────────────────────────────────────────────────────────────
print("\n-- Churn label validation (means by churn status) --")
check = ["days_since_last_trip", "trips_last_30d", "trips_last_90d", "monetary_total"]
print(df.groupby("churned")[check].mean().round(2))
print("\nExpected: churned=1 should have MORE days_since_last_trip and FEWER trips")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(PROC, "features.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df):,} rows -> {out_path}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
