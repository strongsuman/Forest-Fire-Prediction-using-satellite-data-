# 🔥 Madhya Pradesh Forest Fire Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-1192EE.svg)](https://xgboost.readthedocs.io/)
[![NASA FIRMS](https://img.shields.io/badge/Dataset-NASA%20FIRMS-red.svg)](https://firms.modaps.eosdis.nasa.gov/)

A state-of-the-art **Machine Learning & Real-time Weather System** designed to predict **Forest Fire Occurrence**, **Fire Radiative Power (FRP)**, and overall **Environmental Hazard Index** across the state of **Madhya Pradesh, India**.

---

## 🌟 Key Features

- 🛰️ **NASA FIRMS Satellite Data Integration**: Trained on MODIS / VIIRS satellite thermal anomaly datasets over Madhya Pradesh.
- 🤖 **Ensemble Machine Learning Model**: Combines **XGBoost Regressor** and **Random Forest** models with spatial-temporal rolling lag features (`frp_rolling_mean_3`, `frp_rolling_max_7`, `prev_frp`, cyclical month/hour encodings).
- 🗺️ **Interactive Spatial Leaflet Map**: Interactive Leaflet.js satellite basemap with click-to-pick coordinates and MP national park markers.
- 🌲 **Forest Reserve Presets**: One-click quick assessment for key national parks and reserves:
  - **Kanha National Park** (Mandla / Balaghat)
  - **Bandhavgarh National Park** (Umaria)
  - **Panna Tiger Reserve** (Panna / Chhatarpur)
  - **Satpura Tiger Reserve** (Hoshangabad)
  - **Pench Tiger Reserve** (Seoni / Chhindwara)
  - **Bhopal, Indore, and Jabalpur Forest Circles**
- ⛅ **Real-time Weather Integration**: Fetches live temperature, humidity, wind speed, and 7-day cumulative rainfall via Open-Meteo API.
- 📈 **Historical FRP Time-Series Trend**: Interactive Chart.js graph displaying historical fire intensity time-series within the spatial grid.
- 🖨️ **Exportable Fire Risk Report**: Instant formatted print/PDF export capabilities for field officers and research analysis.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   NASA FIRMS Satellite Data   │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│     Live Weather API          │  │ Spatial & Temporal Feature    │
│    (Open-Meteo Forecast)      │  │ Engineering & Lag Features    │
└──────────────┬────────────────┘  └──────────────┬────────────────┘
               │                                  │
               └─────────────────┬────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │ XGBoost + Random Forest Ensemble ML  │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │      FastAPI REST Server (Port 8000) │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   Glassmorphic Interactive Web UI    │
              │  (Leaflet Map + Chart.js Analytics)  │
              └──────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
project/
├── app.py                  # FastAPI Backend API Server & Static Web App host
├── frontend.html           # Modern Glassmorphic Web Dashboard with Leaflet Map
├── run.py                  # Automated Launcher Script (starts API & opens browser)
├── requirements.txt        # Python package dependencies
├── model.pkl               # Trained XGBoost Regressor model
├── rf_model.pkl            # Trained Random Forest Regressor model
├── scaler.pkl              # StandardScaler artifact
├── features.pkl            # List of model input feature columns
├── thresholds.pkl          # Empirical risk quantile thresholds (q1, q2)
├── ensemble_weight.pkl     # Optimal ensemble weighting factor
├── README.md               # Project documentation
└── MP_fire_dataset/        # NASA FIRMS satellite CSV datasets
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Run the automated launcher script:
```bash
python run.py
```

This will automatically start the FastAPI backend server on `http://127.0.0.1:8050` and open the web dashboard in your default browser.

Alternatively, launch Uvicorn directly:
```bash
uvicorn app:app --port 8000 --reload
```

---

## 🔗 API Documentation Reference

### `GET /`
Serves the web dashboard (`frontend.html`).

### `GET /api/health`
Returns system health, model configuration, dataset record count, and spatial bounding box.

### `GET /api/presets`
Returns list of preset Madhya Pradesh national parks and forest circles with spatial coordinates.

### `POST /predict`
Predicts Fire Radiative Power (FRP), Fire Risk Level, and hazard breakdown factors.
```json
// Request Body
{
  "latitude": 22.3345,
  "longitude": 80.6115,
  "daynight": 1
}

// Response
{
  "location": "22.3345° N, 80.6115° E",
  "frp_prediction": 14.85,
  "fire_risk_level": "Medium",
  "nearest_actual_frp": 16.2,
  "nearest_distance_km": 2.4,
  "weather": {
    "temperature": 34.2,
    "humidity": 32.0,
    "wind_speed": 14.5,
    "rain_7days": 0.0
  },
  "risk_factors": {
    "overall_hazard_index": 54.2,
    "temperature_risk_pct": 56.8,
    "humidity_risk_pct": 72.0,
    "wind_risk_pct": 50.8,
    "frp_intensity_pct": 54.7
  }
}
```

### `POST /trend`
Retrieves historical FRP time-series for specified spatial coordinates.

---

## 👤 Author & Acknowledgments

- **Developer**: Jyoti Suman
- **Data Source**: NASA FIRMS (Fire Information for Resource Management System)
- **Weather API**: Open-Meteo Weather Forecast API
