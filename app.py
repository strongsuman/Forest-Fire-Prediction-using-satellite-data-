import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
import asyncio
import joblib
import httpx
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Madhya Pradesh Forest Fire Prediction API",
    description="Machine Learning & Live Weather API for Forest Fire Risk Prediction in Madhya Pradesh",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Model Artifacts ────────────────────────────────────────────────────────
print("[INFO] Loading Machine Learning model artifacts...")
try:
    model      = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    rf_model   = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
    scaler     = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    features   = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    thresholds = joblib.load(os.path.join(BASE_DIR, "thresholds.pkl"))
    best_w     = joblib.load(os.path.join(BASE_DIR, "ensemble_weight.pkl"))
    q1, q2     = thresholds["q1"], thresholds["q2"]
    print("[SUCCESS] Model artifacts loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")
    raise e

# ── 2. Dataset Auto-Discovery & Vectorized Feature Prep ───────────────────────
candidate_paths = [
    os.path.join(BASE_DIR, "..", "MP_fire_dataset", "*.csv"),
    os.path.join(BASE_DIR, "MP_fire_dataset", "*.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "MP_fire_dataset", "*.csv"),
    "e:/Machine Learning/MP_fire_dataset/*.csv"
]

dataset_files = []
for cp in candidate_paths:
    matched = glob.glob(cp)
    if matched:
        dataset_files = matched
        break

if not dataset_files:
    print("[WARNING] Could not locate MP_fire_dataset CSV files in candidate paths.")
    df_raw = pd.DataFrame()
else:
    print(f"[INFO] Found dataset files: {dataset_files}")
    dfs = []
    for f in dataset_files:
        try:
            dfs.append(pd.read_csv(f, encoding="latin1"))
        except Exception as ex:
            print(f"[WARNING] Error reading {f}: {ex}")
    
    if dfs:
        df_raw = pd.concat(dfs, ignore_index=True)
        df_raw['acq_date'] = pd.to_datetime(df_raw['acq_date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['acq_date', 'frp'])
        df_raw = df_raw[df_raw['frp'] > 0]
        df_raw = df_raw.sort_values(['latitude', 'longitude', 'acq_date']).reset_index(drop=True)

        df_raw['daynight'] = df_raw['daynight'].map({'D': 1, 'N': 0}).fillna(1).astype(int)
        df_raw['acq_time'] = df_raw['acq_time'].astype(str).str.zfill(4)
        df_raw['hour']     = pd.to_numeric(df_raw['acq_time'].str[:2], errors='coerce').fillna(12).astype(int)
        df_raw.drop(columns=['acq_time', 'satellite', 'instrument', 'version', 'type'], errors='ignore', inplace=True)
        df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

        med = df_raw['frp'].median()
        df_raw['prev_frp'] = df_raw.groupby(['latitude', 'longitude'])['frp'].shift(1).fillna(med)
        df_raw['frp_rolling_mean_3'] = df_raw['prev_frp'].rolling(3, min_periods=1).mean().fillna(med)
        df_raw['frp_rolling_max_7']  = df_raw['prev_frp'].rolling(7, min_periods=1).max().fillna(med)
        print(f"[SUCCESS] Dataset processed: {df_raw.shape[0]} fire observation records.")
    else:
        df_raw = pd.DataFrame()

# ── 3. Spatial Nearest Lookup & Fast Trend Caching ─────────────────────────────
_lats = df_raw['latitude'].to_numpy() if not df_raw.empty else np.array([])
_lons = df_raw['longitude'].to_numpy() if not df_raw.empty else np.array([])
_trend_cache: Dict[tuple, Dict[str, Any]] = {}

def _build_trend_cache():
    if df_raw.empty:
        return
    print("[INFO] Building spatial trend cache...")
    df_raw['_rlat'] = (_lats * 2).round() / 2   # 0.5 degree grid
    df_raw['_rlon'] = (_lons * 2).round() / 2
    df_raw['_date_str'] = df_raw['acq_date'].dt.strftime('%Y-%m-%d')
    for (rlat, rlon), grp in df_raw.groupby(['_rlat', '_rlon']):
        daily = (grp.groupby('_date_str')['frp']
                 .median().reset_index().sort_values('_date_str'))
        if len(daily) > 60:
            step = max(1, len(daily) // 60)
            daily = daily.iloc[::step]
        _trend_cache[(round(float(rlat), 1), round(float(rlon), 1))] = {
            "dates":       daily['_date_str'].tolist(),
            "frp_values":  [round(v, 2) for v in daily['frp'].tolist()],
            "point_count": len(grp),
        }
    df_raw.drop(columns=['_rlat', '_rlon', '_date_str'], inplace=True, errors='ignore')
    print(f"[SUCCESS] Fast trend cache built: {len(_trend_cache)} spatial grid cells.")

_build_trend_cache()

# ── 4. Presets Data ───────────────────────────────────────────────────────────
MP_PRESETS = [
    {
        "id": "kanha",
        "name": "Kanha National Park",
        "category": "Tiger Reserve & Core Forest",
        "latitude": 22.3345,
        "longitude": 80.6115,
        "district": "Mandla / Balaghat",
        "description": "Dense sal and bamboo forests, highly prone to seasonal dry leaf litter fires."
    },
    {
        "id": "bandhavgarh",
        "name": "Bandhavgarh National Park",
        "category": "National Park & Wildlife Sanctuary",
        "latitude": 23.7019,
        "longitude": 81.0264,
        "district": "Umaria",
        "description": "Tropical moist deciduous forest with steep hill topography and dense vegetation."
    },
    {
        "id": "panna",
        "name": "Panna Tiger Reserve",
        "category": "Tiger Reserve & Biosphere",
        "latitude": 24.5800,
        "longitude": 80.0500,
        "district": "Panna / Chhatarpur",
        "description": "Dry teak forests and riverine habitats along Ken River, susceptible to summer fires."
    },
    {
        "id": "satpura",
        "name": "Satpura Tiger Reserve",
        "category": "Mountainous Forest Reserve",
        "latitude": 22.4833,
        "longitude": 78.4333,
        "district": "Hoshangabad (Narmadapuram)",
        "description": "Rugged sandstone terrain with teak, bamboo, and medicinal flora."
    },
    {
        "id": "pench",
        "name": "Pench Tiger Reserve",
        "category": "Deciduous Teak Forest",
        "latitude": 21.7333,
        "longitude": 79.3167,
        "district": "Seoni / Chhindwara",
        "description": "Southern tropical dry deciduous teak forest bordering Maharashtra."
    },
    {
        "id": "bhopal_circle",
        "name": "Bhopal Forest Circle",
        "category": "Capital Region Forest Range",
        "latitude": 23.2599,
        "longitude": 77.4126,
        "district": "Bhopal",
        "description": "Urban-adjacent forest patches including Van Vihar & Catchment Area."
    },
    {
        "id": "indore_circle",
        "name": "Indore Forest Division",
        "category": "Western MP Plateau",
        "latitude": 22.7196,
        "longitude": 75.8577,
        "district": "Indore",
        "description": "Malwa plateau dry deciduous vegetation and river basins."
    },
    {
        "id": "jabalpur_circle",
        "name": "Jabalpur Forest Division",
        "category": "Central MP Forest Belt",
        "latitude": 23.1815,
        "longitude": 79.9864,
        "district": "Jabalpur",
        "description": "Narmada valley teak & mixed deciduous forest stretches."
    }
]

# ── 5. Schemas ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    latitude:  float = Field(..., ge=17.0, le=27.0, description="Latitude inside Madhya Pradesh (17.0 - 27.0)")
    longitude: float = Field(..., ge=74.0, le=85.0, description="Longitude inside Madhya Pradesh (74.0 - 85.0)")
    daynight:  int   = Field(1, description="Observation time: 1 for Daytime, 0 for Nighttime")

class TrendRequest(BaseModel):
    latitude:  float = Field(..., ge=17.0, le=27.0)
    longitude: float = Field(..., ge=74.0, le=85.0)
    radius:    float = Field(0.5, description="Search radius in degrees")

# ── 6. Helpers ────────────────────────────────────────────────────────────────
def get_nearest_row(lat: float, lon: float) -> pd.Series:
    if len(_lats) == 0:
        return pd.Series({
            'latitude': lat, 'longitude': lon, 'hour': 12, 'acq_date': pd.Timestamp.now(),
            'brightness': 310.0, 'confidence': 75.0, 'scan': 1.0, 'track': 1.0,
            'bright_t31': 295.0, 'prev_frp': 10.0, 'frp_rolling_mean_3': 12.0,
            'frp_rolling_max_7': 25.0, 'frp': 15.0
        })
    dists = (_lats - lat)**2 + (_lons - lon)**2
    return df_raw.iloc[int(np.argmin(dists))]

def categorize(pred: float) -> str:
    if pred < q1:   return "Low"
    elif pred < q2: return "Medium"
    else:           return "High"

SEASON_MAP = {12:'winter', 1:'winter', 2:'winter',
               3:'spring', 4:'spring', 5:'spring',
               6:'summer', 7:'summer', 8:'summer',
               9:'autumn', 10:'autumn', 11:'autumn'}

def build_feature_row(lat: float, lon: float, daynight: int, row: pd.Series) -> pd.DataFrame:
    hour   = int(row['hour'])
    month  = int(row['acq_date'].month) if isinstance(row['acq_date'], pd.Timestamp) else 5
    dayofyear = int(row['acq_date'].dayofyear) if isinstance(row['acq_date'], pd.Timestamp) else 135
    season = SEASON_MAP.get(month, 'spring')
    
    feat = {
        'latitude':              lat,
        'longitude':             lon,
        'daynight':              daynight,
        'brightness':            float(row['brightness']),
        'confidence':            float(row['confidence']),
        'scan':                  float(row['scan']),
        'track':                 float(row['track']),
        'bright_t31':            float(row['bright_t31']),
        'hour':                  hour,
        'month':                 month,
        'dayofyear':             dayofyear,
        'hour_sin':              np.sin(2*np.pi*hour/24),
        'hour_cos':              np.cos(2*np.pi*hour/24),
        'month_sin':             np.sin(2*np.pi*month/12),
        'month_cos':             np.cos(2*np.pi*month/12),
        'brightness_confidence': float(row['brightness']) * float(row['confidence']),
        'prev_frp':              float(row['prev_frp']),
        'frp_rolling_mean_3':    float(row['frp_rolling_mean_3']),
        'frp_rolling_max_7':     float(row['frp_rolling_max_7']),
        'season_spring':         1 if season=='spring' else 0,
        'season_summer':         1 if season=='summer' else 0,
        'season_winter':         1 if season=='winter' else 0,
    }
    df_feat = pd.DataFrame([feat])
    for col in features:
        if col not in df_feat.columns:
            df_feat[col] = 0
    return df_feat[features]

async def get_weather(lat: float, lon: float) -> dict:
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
           f"&daily=precipitation_sum&past_days=7&forecast_days=1&timezone=Asia/Kolkata")
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                d = r.json()
                return {
                    "temperature": d["current"]["temperature_2m"],
                    "humidity":    d["current"]["relative_humidity_2m"],
                    "wind_speed":  d["current"]["wind_speed_10m"],
                    "rain_7days":  round(sum(v or 0 for v in d.get("daily", {}).get("precipitation_sum", [])), 1),
                }
    except Exception:
        pass
    return {
        "temperature": 32.5,
        "humidity": 38.0,
        "wind_speed": 12.0,
        "rain_7days": 0.0,
        "is_fallback": True
    }

def compute_risk_factors(pred_frp: float, weather: dict, prev_frp: float) -> dict:
    temp = weather.get("temperature", 30.0)
    temp_score = min(100.0, max(0.0, (temp - 20.0) * 4.0))
    
    hum = weather.get("humidity", 40.0)
    hum_score = min(100.0, max(0.0, (80.0 - hum) * 1.5))
    
    wind = weather.get("wind_speed", 10.0)
    wind_score = min(100.0, max(0.0, wind * 3.5))
    
    frp_score = min(100.0, max(0.0, (pred_frp / (q2 * 1.5)) * 100.0))
    
    overall_index = round(0.4 * frp_score + 0.25 * temp_score + 0.2 * hum_score + 0.15 * wind_score, 1)
    
    return {
        "overall_hazard_index": overall_index,
        "temperature_risk_pct": round(temp_score, 1),
        "humidity_risk_pct": round(hum_score, 1),
        "wind_risk_pct": round(wind_score, 1),
        "frp_intensity_pct": round(frp_score, 1),
    }

# ── 7. Routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the interactive web user interface."""
    frontend_path = os.path.join(BASE_DIR, "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Forest Fire Prediction API is running. frontend.html not found."}

@app.get("/api/health")
def health_check():
    """Returns system status, model details, and dataset metadata."""
    data_date_min = str(df_raw['acq_date'].min().date()) if not df_raw.empty else "N/A"
    data_date_max = str(df_raw['acq_date'].max().date()) if not df_raw.empty else "N/A"
    return {
        "status": "healthy",
        "api_name": "Madhya Pradesh Forest Fire Prediction System",
        "version": "2.0.0",
        "dataset_info": {
            "total_records": len(df_raw),
            "spatial_coverage": "Madhya Pradesh (17°N - 27°N, 74°E - 85°E)",
            "date_range": f"{data_date_min} to {data_date_max}",
            "cached_grid_cells": len(_trend_cache),
        },
        "model_info": {
            "ensemble_algorithm": "XGBoost + Random Forest",
            "xgb_weight": round(float(best_w), 2),
            "rf_weight": round(1.0 - float(best_w), 2),
            "thresholds": {"q1_low_max": round(q1, 2), "q2_medium_max": round(q2, 2)},
            "features_count": len(features),
        }
    }

@app.get("/api/presets")
def get_presets():
    """Returns preset MP Forest Reserves & National Parks for quick UI selection."""
    return {"presets": MP_PRESETS}

@app.post("/predict")
async def predict(data: PredictRequest):
    """Predicts fire radiative power (FRP), fire risk level, and environmental hazard factors."""
    lat, lon, daynight = data.latitude, data.longitude, data.daynight

    if not (17.0 <= lat <= 27.0 and 74.0 <= lon <= 85.0):
        raise HTTPException(
            status_code=400,
            detail="Coordinates outside Madhya Pradesh boundary (Latitude 17-27°N, Longitude 74-85°E)."
        )

    row          = get_nearest_row(lat, lon)
    input_df     = build_feature_row(lat, lon, daynight, row)
    input_scaled = scaler.transform(input_df)

    xgb_p = float(np.expm1(model.predict(input_scaled)[0]))
    rf_p  = float(rf_model.predict(input_df)[0])
    pred  = float(best_w * xgb_p + (1 - best_w) * rf_p)
    dist  = float(np.sqrt((float(row['latitude'])-lat)**2 + (float(row['longitude'])-lon)**2))

    weather = await get_weather(lat, lon)
    risk_factors = compute_risk_factors(pred, weather, float(row['prev_frp']))
    risk_level = categorize(pred)

    data_date_str = str(row['acq_date'].date()) if isinstance(row['acq_date'], pd.Timestamp) else str(pd.Timestamp.now().date())

    return {
        "location":             f"{lat:.4f}° N, {lon:.4f}° E",
        "latitude":             lat,
        "longitude":            lon,
        "frp_prediction":       round(pred, 2),
        "fire_risk_level":      risk_level,
        "nearest_actual_frp":   round(float(row['frp']), 2),
        "nearest_distance_deg": round(dist, 4),
        "nearest_distance_km":  round(dist * 111.0, 1),
        "data_date":            data_date_str,
        "thresholds":           {"low_max": round(q1, 2), "medium_max": round(q2, 2)},
        "weather":              weather,
        "risk_factors":         risk_factors,
    }

@app.post("/trend")
def trend(data: TrendRequest):
    """Retrieves spatial historical FRP trend time-series for specified location."""
    lat, lon = data.latitude, data.longitude

    rlat = round(round(lat * 2) / 2, 1)
    rlon = round(round(lon * 2) / 2, 1)

    result = _trend_cache.get((rlat, rlon))
    if result is None:
        best_dist, best_key = float('inf'), None
        for (clat, clon) in _trend_cache:
            d = (clat - lat)**2 + (clon - lon)**2
            if d < best_dist:
                best_dist, best_key = d, (clat, clon)
        if best_key:
            result = _trend_cache.get(best_key, {})

    if not result or not result.get("dates"):
        raise HTTPException(status_code=404, detail="No historical trend data found for this location.")

    return {
        "dates":       result["dates"],
        "frp_values":  result["frp_values"],
        "point_count": result["point_count"],
        "radius_deg":  0.5,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=True)