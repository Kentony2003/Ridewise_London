# =============================================================================
# RideWise --- Exploratory Data Analysis (EDA)
# =============================================================================


# =============================================================================
# STEP 1 - Setup & Imports
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False
})


# =============================================================================
# STEP 2 - Loading the Data
# =============================================================================

base_path = r"C:/Users/THINKPAD/OneDrive/Desktop/DATA FILES/Ridewise_London/data/raw/"

riders     = pd.read_csv(base_path + "riders.csv",     parse_dates=["signup_date"])
trips      = pd.read_csv(base_path + "trips.csv")
drivers    = pd.read_csv(base_path + "drivers.csv",    parse_dates=["signup_date"])
sessions   = pd.read_csv(base_path + "sessions.csv")
promotions = pd.read_csv(base_path + "promotions.csv")


# =============================================================================
# STEP 2a - Data Overview
# =============================================================================

for name, df in [("riders", riders), ("trips", trips), ("drivers", drivers),
                 ("sessions", sessions), ("promotions", promotions)]:
    print(f" {name:<12} {df.shape[0]:>7} rows {df.shape[1]:>2} cols")


# =============================================================================
# STEP 2b - Data Inspection
# =============================================================================

print(riders.dtypes)
print(riders.isnull().sum())
print(riders.head())


# =============================================================================
# STEP 3 - Visualization of the Riders (Data Distribution)
# =============================================================================

# Age distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Step 3b - Age distribution
axes[0].hist(riders["age"].dropna(), bins=30, color="skyblue", edgecolor="black")
axes[0].axvline(
    riders["age"].mean(), color="red", linestyle="dashed",
    label=f"Mean: {riders['age'].mean():.1f}"
)
axes[0].set(title="Age Distribution of Riders", xlabel="Age", ylabel="Riders")
axes[0].legend()

# Step 3d - City Distribution
city_counts = riders["city"].value_counts()
axes[1].bar(city_counts.index, city_counts.values, color="lightblue", edgecolor="black")
for i, v in enumerate(city_counts.values):
    axes[1].text(i, v + 30, f"{v:,}", ha="center")
axes[1].set(title="Riders by City", xlabel="City", ylabel="Number of Riders")

# Step 3g - Loyalty Distribution
loyalty_counts = riders["loyalty_status"].value_counts()
axes[2].bar(
    loyalty_counts.index, loyalty_counts.values,
    color=["lightcoral", "lightgreen", "lightblue"], edgecolor="black"
)
for i, v in enumerate(loyalty_counts.values):
    axes[2].text(i, v + 30, f"{v:,}", ha="center")
axes[2].set(title="Riders by Loyalty", xlabel="Loyalty Status", ylabel="Number of Riders")

plt.suptitle("Rider Demographics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 4 - Churn Probability Column
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(riders["churn_prob"].dropna(), bins=30, color="skyblue", edgecolor="black")
ax.set(title="Churn Probability Distribution of Riders", xlabel="Churn Probability", ylabel="Riders")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 5 - Sign Up Trends Over Time
# =============================================================================

monthly = riders.set_index("signup_date").resample("ME").size().reset_index(name="signups")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly["signup_date"], monthly["signups"], marker="o", color="skyblue")
ax.set(title="Monthly Signups Over Time", xlabel="Signup Date", ylabel="Number of Signups")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6 - Trips Analysis
# =============================================================================

print(trips.head())
print(f"Shape : {trips.shape}")
print(trips.dtypes)
print(trips.isnull().sum())


# =============================================================================
# STEP 6a - Convert to Proper Date Format
# =============================================================================

for col in ["pickup_time", "dropoff_time"]:
    trips[col] = pd.to_datetime(trips[col], utc=True, errors="coerce").dt.tz_localize(None)


# =============================================================================
# STEP 6b - Engineer Trip Features
# =============================================================================

# 1 - Trip duration in minutes
trips["trip_duration_min"] = (
    (trips["dropoff_time"] - trips["pickup_time"]).dt.total_seconds() / 60
)

# 2 - Total revenue: fare x surge_multiplier + tip_amount
trips["total_revenue"] = trips["fare"] * trips["surge_multiplier"] + trips["tip"].fillna(0)

# 3 - Hour of the day and day of the week
trips["hour_of_day"] = trips["pickup_time"].dt.hour
trips["day_of_week"] = trips["pickup_time"].dt.day_name()


# =============================================================================
# STEP 6c - Financial & Duration Distribution
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, col, title, color in zip(
    axes,
    ["fare", "surge_multiplier", "trip_duration_min"],
    ["Fare(EUR)", "Surge Multiplier", "Duration(min)"],
    ["green", "yellow", "blue"]
):
    data = trips[col].clip(upper=trips[col].quantile(0.99)).dropna()
    ax.hist(data, bins=35, color=color, edgecolor="black")
    ax.axvline(
        data.mean(), color="black", linestyle="dashed",
        label=f"Mean: {data.mean():.2f}"
    )
    ax.set(title=title, xlabel=title, ylabel="Number of Trips")
    ax.legend()

plt.suptitle("Trips Financials & Duration", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6d - Temporal Patterns: When Do People Ride?
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Hour of the day
hour_counts = trips["hour_of_day"].value_counts().sort_index()
axes[0].bar(hour_counts.index, hour_counts.values, color="skyblue", edgecolor="black")
for h in [7, 8, 9, 17, 18, 19]:
    axes[0].axvspan(h - 0.5, h + 0.5, alpha=0.15, color="orange")
axes[0].set(title="Trips by Hour of the Day", xlabel="Hour", ylabel="Trips")
axes[0].set_xticks(range(0, 24, 2))

# Day of week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = trips["day_of_week"].value_counts().reindex(day_order)
axes[1].bar(
    day_order, day_counts.values,
    color=["lightgreen"] * 5 + ["lightcoral"] * 2, edgecolor="black"
)
axes[1].set(title="Trips by Day of the Week", xlabel="Day of Week", ylabel="Trips")
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("Trip Timing Patterns", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6e - Payment & Weather Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Payment type distribution
payment_counts = trips["payment_type"].value_counts()
axes[0].pie(
    payment_counts.values, labels=payment_counts.index, autopct="%1.1f%%",
    colors=sns.color_palette("husl", len(payment_counts)),
    wedgeprops={"edgecolor": "black"}
)
axes[0].set(title="Payment Type Distribution")

# Weather distribution
weather_counts = trips["weather"].value_counts()
axes[1].bar(
    weather_counts.index, weather_counts.values,
    color=sns.color_palette("Set2", len(weather_counts)), edgecolor="black"
)
axes[1].set(title="Weather Distribution")

plt.suptitle("Trip Conditions", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 7 - Drivers Dataset Analysis
# =============================================================================

print(drivers.head())
print(f"Shape : {drivers.shape}")
print(drivers.dtypes)
print(drivers.isnull().sum())


# =============================================================================
# STEP 7a - Driver Fleet Composition
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Vehicle type
vehicle_counts = drivers["vehicle_type"].value_counts()
axes[0].bar(
    vehicle_counts.index, vehicle_counts.values,
    color=sns.color_palette("Set2", len(vehicle_counts)), edgecolor="white"
)
axes[0].set(title="Vehicle Types", xlabel="Type", ylabel="")

# Rating distribution
axes[1].hist(drivers["rating"].dropna(), bins=20, color="skyblue", edgecolor="black")
axes[1].axvline(4.0, color="red", linestyle="dashed", label="4.0 Threshold")
axes[1].set(title="Driver Ratings", xlabel="Rating", ylabel="Drivers")
axes[1].legend()

plt.suptitle("Driver Characteristics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 8 - Session Dataset Analysis
# =============================================================================

print(sessions.head())
print(f"Shape : {sessions.shape}")
print(sessions.dtypes)
print(sessions.isnull().sum())

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Session duration distribution
axes[0].hist(
    sessions["time_on_app"].clip(upper=sessions["time_on_app"].quantile(0.99)).dropna(),
    bins=30, color="skyblue", edgecolor="black"
)
axes[0].set(title="Session Duration Distribution", xlabel="Time on App (min)", ylabel="Sessions")

# Conversion rate by loyalty status
conv_loyalty = sessions.groupby("loyalty_status")["converted"].mean().mul(100)
axes[1].bar(conv_loyalty.index, conv_loyalty.values, edgecolor="black")
for i, v in enumerate(conv_loyalty.values):
    axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center")
axes[1].set(title="Conversion Rate by Loyalty Status", xlabel="Loyalty Status", ylabel="Conversion Rate")

plt.suptitle("Session Insights", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 9 - Churn Analysis
# =============================================================================

# Engineer a binary "churned" column using 0.5 threshold
riders["churned"] = (riders["churn_prob"] >= 0.5).astype(int)

print(f"Overall churn rate : {riders['churned'].mean() * 100:.1f}%")
print(f"Churned            : {riders['churned'].sum()}")


# =============================================================================
# STEP 9c - Churn by Loyalty & City (Bi-variate Analysis)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Churn by loyalty
churn_loyalty = riders.groupby("loyalty_status")["churned"].mean().mul(100)
axes[0].bar(churn_loyalty.index, churn_loyalty.values, edgecolor="black")
for i, v in enumerate(churn_loyalty.values):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center")
axes[0].set(title="Churn Rate by Loyalty Status", xlabel="Loyalty Status", ylabel="Churn Rate (%)")

# Churn by city
churn_city = riders.groupby("city")["churned"].mean().mul(100).sort_values(ascending=False)
overall = riders["churned"].mean() * 100
axes[1].bar(churn_city.index, churn_city.values, edgecolor="black")
axes[1].axhline(overall, color="red", linestyle="dashed", label=f"Overall: {overall:.1f}%")
for i, v in enumerate(churn_city.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center")
axes[1].set(title="Churn Rate by City", xlabel="City", ylabel="Churn Rate (%)")
axes[1].legend()

plt.suptitle("Rider Churn Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 9d - Churn Probability Distribution (Final)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(riders["churn_prob"].dropna(), bins=30, color="skyblue", edgecolor="black")
ax.axvline(0.5, color="red", linestyle="dashed", label="Churn Threshold (0.5)")
ax.set(title="Churn Probability Distribution of Riders", xlabel="Churn Probability", ylabel="Riders")
ax.legend()
plt.tight_layout()
plt.show()