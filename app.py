# -------------------------------
# IMPORTS
# -------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# -------------------------------
# INIT APP
# -------------------------------
app = FastAPI(title="🔥 Forest Fire Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD MODEL ARTIFACTS (once at startup)
# -------------------------------
model         = joblib.load("model.pkl")          # XGBoost
rf_model      = joblib.load("rf_model.pkl")       # Random Forest
scaler        = joblib.load("scaler.pkl")
features      = joblib.load("features.pkl")
thresholds    = joblib.load("thresholds.pkl")
best_w        = joblib.load("ensemble_weight.pkl")

q1 = thresholds["q1"]
q2 = thresholds["q2"]

print("✅ All model artifacts loaded")

# -------------------------------
# LOAD DATASET — for nearest-point lookup
# FIX: sort by location+date to match training sort order
# -------------------------------
files = glob.glob("../MP_fire_dataset/*.csv")
df_raw = pd.concat([pd.read_csv(f, encoding="latin1") for f in files], ignore_index=True)

df_raw['acq_date'] = pd.to_datetime(df_raw['acq_date'], errors='coerce')
df_raw = df_raw.dropna(subset=['acq_date', 'frp'])
df_raw = df_raw[df_raw['frp'] > 0]

# Sort exactly as in training
df_raw = df_raw.sort_values(['latitude', 'longitude', 'acq_date']).reset_index(drop=True)

df_raw['daynight'] = df_raw['daynight'].map({'D': 1, 'N': 0}).fillna(1).astype(int)
df_raw['acq_time'] = df_raw['acq_time'].astype(str).str.zfill(4)
df_raw['hour']     = pd.to_numeric(df_raw['acq_time'].str[:2], errors='coerce').fillna(12).astype(int)
df_raw.drop(columns=['acq_time','satellite','instrument','version','type'],
            errors='ignore', inplace=True)
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

# Pre-compute lag/rolling features on the full dataset lookup table
df_raw['prev_frp'] = (
    df_raw.groupby(['latitude', 'longitude'])['frp']
    .shift(1)
    .fillna(df_raw['frp'].median())
)
df_raw['frp_rolling_mean_3'] = (
    df_raw.groupby(['latitude', 'longitude'])['frp']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .fillna(df_raw['frp'].median())
)
df_raw['frp_rolling_max_7'] = (
    df_raw.groupby(['latitude', 'longitude'])['frp']
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    .fillna(df_raw['frp'].median())
)

# Dataset-level medians (fallbacks)
MEDIAN_PREV_FRP           = float(df_raw['frp'].median())
MEDIAN_ROLLING_MEAN_3     = float(df_raw['frp_rolling_mean_3'].median())
MEDIAN_ROLLING_MAX_7      = float(df_raw['frp_rolling_max_7'].median())

print("✅ Dataset loaded:", df_raw.shape)

# -------------------------------
# REQUEST SCHEMA
# -------------------------------
class PredictRequest(BaseModel):
    latitude:  float
    longitude: float
    daynight:  int = 1     # 1=Day, 0=Night (optional, defaults to Day)

# -------------------------------
# HELPERS
# -------------------------------
def get_nearest_row(lat: float, lon: float) -> pd.Series:
    """Return the most recent satellite observation nearest to (lat, lon)."""
    temp = df_raw.copy()
    temp['_dist'] = np.sqrt(
        (temp['latitude']  - lat) ** 2 +
        (temp['longitude'] - lon) ** 2
    )
    # Among nearest candidates, take the most recent one
    nearest = (
        temp.sort_values(['_dist', 'acq_date'], ascending=[True, False])
        .iloc[0]
    )
    return nearest


def categorize(pred: float) -> str:
    if pred < q1:
        return "Low"
    elif pred < q2:
        return "Medium"
    else:
        return "High"


def build_feature_row(lat, lon, daynight, row) -> pd.DataFrame:
    """Build the full feature DataFrame for a single prediction."""
    brightness  = float(row['brightness'])
    confidence  = float(row['confidence'])
    scan        = float(row['scan'])
    track       = float(row['track'])
    bright_t31  = float(row['bright_t31'])
    hour        = int(row['hour'])
    month       = int(row['acq_date'].month)
    dayofyear   = int(row['acq_date'].dayofyear)
    prev_frp    = float(row['prev_frp'])
    rolling3    = float(row['frp_rolling_mean_3'])
    rolling7    = float(row['frp_rolling_max_7'])

    season_map = {12:'winter',1:'winter',2:'winter',
                  3:'spring',4:'spring',5:'spring',
                  6:'summer',7:'summer',8:'summer',
                  9:'autumn',10:'autumn',11:'autumn'}
    season = season_map[month]

    feat = {
        'latitude':              lat,
        'longitude':             lon,
        'daynight':              daynight,
        'brightness':            brightness,
        'confidence':            confidence,
        'scan':                  scan,
        'track':                 track,
        'bright_t31':            bright_t31,
        'hour':                  hour,
        'month':                 month,
        'dayofyear':             dayofyear,
        'hour_sin':              np.sin(2 * np.pi * hour  / 24),
        'hour_cos':              np.cos(2 * np.pi * hour  / 24),
        'month_sin':             np.sin(2 * np.pi * month / 12),
        'month_cos':             np.cos(2 * np.pi * month / 12),
        'brightness_confidence': brightness * confidence,
        'prev_frp':              prev_frp,
        'frp_rolling_mean_3':    rolling3,
        'frp_rolling_max_7':     rolling7,
    }

    # Season dummies (autumn is the dropped base)
    for s in ['spring', 'summer', 'winter']:
        feat[f'season_{s}'] = 1 if season == s else 0

    df_feat = pd.DataFrame([feat])

    # Ensure all training columns exist
    for col in features:
        if col not in df_feat.columns:
            df_feat[col] = 0

    return df_feat[features]


# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"message": "🔥 Fire Prediction API is running"}


@app.post("/predict")
def predict(data: PredictRequest):
    lat      = data.latitude
    lon      = data.longitude
    daynight = data.daynight

    # Validate coordinates (rough bounds for Madhya Pradesh)
    if not (17.0 <= lat <= 27.0 and 74.0 <= lon <= 85.0):
        raise HTTPException(
            status_code=400,
            detail="Coordinates appear to be outside Madhya Pradesh region."
        )

    # Get nearest historical satellite observation
    row = get_nearest_row(lat, lon)

    # Build feature vector
    input_df     = build_feature_row(lat, lon, daynight, row)
    input_scaled = scaler.transform(input_df)

    # Ensemble: XGBoost (log scale) + Random Forest
    xgb_pred_log = model.predict(input_scaled)[0]
    xgb_pred     = float(np.expm1(xgb_pred_log))
    rf_pred      = float(rf_model.predict(input_df)[0])

    prediction   = best_w * xgb_pred + (1 - best_w) * rf_pred

    category     = categorize(prediction)
    nearest_dist = float(np.sqrt(
        (row['latitude'] - lat) ** 2 +
        (row['longitude'] - lon) ** 2
    ))

    return {
        "location":           f"{lat:.4f}, {lon:.4f}",
        "frp_prediction":     round(prediction, 2),
        "fire_risk_level":    category,
        "nearest_actual_frp": round(float(row['frp']), 2),
        "nearest_distance_deg": round(nearest_dist, 4),
        "data_date":          str(row['acq_date'].date()),
        "thresholds":         {"low_max": round(q1, 2),
                               "medium_max": round(q2, 2)},
    }