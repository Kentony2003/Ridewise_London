from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import joblib
import os
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHURN_MODEL_PATH  = os.path.join(MODEL_DIR, "churn_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")
THRESHOLD_PATH    = os.path.join(MODEL_DIR, "optimal_threshold.pkl")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_segmentation.pkl")
SCALER_PATH       = os.path.join(MODEL_DIR, "scalar_segmentation.pkl")
SEG_FEATURES_PATH = os.path.join(MODEL_DIR, "segmentation_features_cols.pkl")

# ── Load Models ───────────────────────────────────────────────────────────────
try:
    with open(CHURN_MODEL_PATH, "rb") as f:
        churn_model = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    with open(THRESHOLD_PATH, "rb") as f:
        optimal_threshold = pickle.load(f)

    kmeans_model     = joblib.load(KMEANS_MODEL_PATH)
    scaler           = joblib.load(SCALER_PATH)
    seg_feature_cols = joblib.load(SEG_FEATURES_PATH)

    print("✅ All models loaded successfully.")

except FileNotFoundError as e:
    raise RuntimeError(f"❌ Model file not found: {e}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ridewise Churn Prediction API",
    description="Predicts the probability of a rider churning based on trip features.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ────────────────────────────────────────────────────────────
class RidewiseFeatures(BaseModel):
    days_since_last_trip:       float = Field(..., example=5,     description="Days since last trip")
    days_since_last_session:    float = Field(..., example=2.0,   description="Days since last app session")
    trips_last_7d:              int   = Field(..., example=1,     description="Trips in last 7 days")
    trips_last_30d:             int   = Field(..., example=4,     description="Trips in last 30 days")
    trips_last_60d:             int   = Field(..., example=8,     description="Trips in last 60 days")
    trips_last_90d:             int   = Field(..., example=12,    description="Trips in last 90 days")
    trips_lifetime:             int   = Field(..., example=50,    description="Total lifetime trips")
    monetary_total:             float = Field(..., example=300.0, description="Total spend (GBP)")
    monetary_avg:               float = Field(..., example=15.0,  description="Average spend per trip (GBP)")
    monetary_last_30d:          float = Field(..., example=60.0,  description="Spend in last 30 days (GBP)")
    rfm_recency_score:          float = Field(..., example=3.0,   description="RFM recency score (1-5)")
    rfm_frequency_score:        float = Field(..., example=3.0,   description="RFM frequency score (1-5)")
    rfm_monetary_score:         float = Field(..., example=3.0,   description="RFM monetary score (1-5)")
    rfm_combined_score:         float = Field(..., example=9.0,   description="Combined RFM score")
    avg_trip_duration:          float = Field(..., example=20.0,  description="Average trip duration (mins)")
    avg_fare:                   float = Field(..., example=15.0,  description="Average fare (GBP)")
    avg_surge:                  float = Field(..., example=0.1,   description="Average surge multiplier")
    tip_rate:                   float = Field(..., example=0.2,   description="Proportion of trips with a tip")
    peak_hour_rate:             float = Field(..., example=0.4,   description="Proportion of trips during peak hours")
    weekend_ratio:              float = Field(..., example=0.3,   description="Proportion of trips on weekends")
    rainy_ride_ratio:           float = Field(..., example=0.1,   description="Proportion of trips in rainy weather")
    preffered_payment_encoded:  int   = Field(..., example=1,     description="Preferred payment method (encoded)")
    avg_driver_rating_received: float = Field(..., example=4.5,   description="Average rating received from drivers")
    avg_rating_given:           float = Field(..., example=4.7,   description="Average rating given to drivers")
    session_last_30d:           float = Field(..., example=5.0,   description="Sessions in last 30 days")
    session_last_60d:           float = Field(..., example=10.0,  description="Sessions in last 60 days")
    avg_time_on_app:            float = Field(..., example=8.0,   description="Average time on app (mins)")
    avg_pages_visited:          float = Field(..., example=3.0,   description="Average pages visited per session")
    session_conversation_rate:  float = Field(..., example=0.25,  description="Session to trip conversion rate")
    engagement_score:           float = Field(..., example=200.0, description="Overall engagement score")
    account_age_days:           int   = Field(..., example=180,   description="Account age in days")
    activity_trend_30d:         float = Field(..., example=0.6,   description="Activity trend over 30 days")
    unique_active_days:         int   = Field(..., example=20,    description="Unique active days")
    was_referred:               int   = Field(..., example=0,     description="1 if rider was referred, 0 otherwise")
    loyalty_encoded:            int   = Field(..., example=0,     description="Loyalty tier (encoded)")
    city_encoded:               int   = Field(..., example=1,     description="City (encoded)")

# ── Response Schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  str
    risk_level:        str

# ── Helper ────────────────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def welcome_root():
    return {
        "message": "Welcome to the Ridewise Churn Prediction API! 🚗",
        "docs":    "Visit /docs for interactive API documentation."
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": churn_model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(rider: RidewiseFeatures):
    try:
        data = pd.DataFrame([rider.dict()])
        data = data[feature_cols]

        if not hasattr(churn_model, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba()")

        churn_prob = round(float(churn_model.predict_proba(data)[0][1]), 4)
        threshold  = float(optimal_threshold) if optimal_threshold is not None else 0.5

        return PredictionResponse(
            churn_probability = churn_prob,
            churn_prediction  = "Yes" if churn_prob >= threshold else "No",
            risk_level        = get_risk_level(churn_prob)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)