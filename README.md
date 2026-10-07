# DROUGHT-X: Multi-Horizon Drought Severity Prediction Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb602b.svg)](https://xgboost.readthedocs.io/)

**DROUGHT-X** is an end-to-end machine learning framework designed for multi-horizon drought severity forecasting across all 254 counties in Texas. By combining historical drought indices (USDM), meteorology time series, spatial neighborhood lag features, and soil characteristics, the system predicts drought category transitions across **1-week, 2-week, 3-week, and 4-week** horizons.

---

## 🌍 Key Capabilities

- **Multi-Horizon Forecasting**: Predicts drought severity across 4 progressive planning horizons ($t+1w$, $t+2w$, $t+3w$, $t+4w$).
- **USDM Classification**: Classifies drought severity according to the official U.S. Drought Monitor standard:
  - **Class 0**: None (No Drought)
  - **Class 1**: D0 (Abnormally Dry)
  - **Class 2**: D1 (Moderate Drought)
  - **Class 3**: D2 (Severe Drought)
  - **Class 4**: D3 (Extreme Drought)
  - **Class 5**: D4 (Exceptional Drought)
- **Spatial Feature Engineering**: Incorporates first-order geographic adjacency, neighbor drought status, and regional weather pressure.
- **Microservices Architecture**: Decoupled FastAPI inference backend with interactive Streamlit visualization dashboard.

---

## 📊 Model Benchmark Summary

| Model | Family | Accuracy | Macro F1 | Weighted F1 | Observation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | Bagged trees | **0.8601** | **0.8038** | **0.8598** | Best Accuracy and Weighted F1; Selected |
| **XGBoost** | Gradient boosting | 0.8574 | 0.7981 | 0.8569 | Fraction of a percentage point from RF |
| **HistGradientBoosting** | Gradient boosting | 0.8566 | 0.7963 | 0.8559 | Fast inference, competitive F1 |
| **Extra Trees** | Randomised trees | 0.8492 | 0.7854 | 0.8488 | Strong ensemble performance |
| **Logistic Regression** | Linear baseline | 0.7820 | 0.6912 | 0.7794 | Baseline linear model |

---

## 🏗️ Repository Architecture

```text
├── app.py                      # Interactive Streamlit frontend dashboard
├── backend/                    # FastAPI backend microservices
│   ├── main.py                 # API endpoints & route handlers
│   ├── prediction_pipeline.py  # End-to-end inference & feature assembly
│   ├── usdm.py                 # Live/historical USDM data retrieval
│   ├── weather.py              # Open-Meteo weather client & rolling aggregations
│   ├── neighbours.py           # Spatial county adjacency & neighbor analysis
│   └── historical.py           # Historical drought trend querying
├── models/                     # Production Random Forest models
│   ├── model_1w.pkl            # 1-week ahead horizon
│   ├── model_2w.pkl            # 2-week ahead horizon
│   ├── model_3w.pkl            # 3-week ahead horizon
│   ├── model_4w.pkl            # 4-week ahead horizon
│   └── experiments/            # Older and experimental model weights
├── data/                       # Production datasets & county geospatial metadata
│   ├── raw/                    # Raw multi-year time-series CSVs & parquets
│   ├── processed/              # Aggregated weekly features & clean datasets
│   ├── county_neighbors.json   # Precomputed Texas county graph adjacency
│   ├── soil_data.csv           # County-level soil composition features
│   ├── texas_county_locations.csv # Latitude/longitude centroids
│   └── texas_geojson.json      # Texas county polygon boundaries
├── notebooks/                  # Jupyter notebooks organized by phase
│   ├── training/               # Data prep & feature engineering pipelines
│   ├── analysis/               # Validation & visualization workflows
│   └── experiments/            # Baseline models & meteorology exploration
├── outputs/                    # Exported predictions, maps & charts
├── docs/                       # Project documentation & architecture
├── tests/                      # Automated system integrity tests
├── _ARCHIVE/                   # Deduplicated historical files & old test stubs
└── _UNRELATED_REVIEW/          # External artifacts flagged for review
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Niranjan2025-CIT/Multi-Horizon-Drought-Severity-Prediction-using-machine-learning.git
cd Multi-Horizon-Drought-Severity-Prediction-using-machine-learning

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The FastAPI documentation will be accessible at: `http://127.0.0.1:8000/docs`.

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the interactive forecasting interface.

---

## 📡 Live API Endpoints

- `GET /health`: Health check and system status.
- `GET /counties`: List of Texas county names, FIPS codes, and coordinates.
- `GET /current-usdm/{fips}`: Fetch latest official USDM drought status for a county.
- `GET /weather/{fips}`: Retrieve 7-day, 14-day, and 30-day meteorological aggregations.
- `GET /predict/{fips}`: Generate multi-horizon forecasts ($t+1w$, $t+2w$, $t+3w$, $t+4w$) with confidence scores.

---

## 🔒 Notes on Large Datasets & Model Weights

In accordance with GitHub file size restrictions (<100 MB per file):
- Large raw time-series CSVs/Parquets (`train_timeseries.csv`, `processed_base.csv`, etc.) are excluded via `.gitignore`.
- Trained Random Forest `.pkl` files (>800 MB each) are excluded from direct git commits.
- All reproducible training scripts and notebooks (`prep.ipynb`, `preprocess.ipynb`, `vis_val.ipynb`) are included in the repository.

---

## 📜 License

This project is licensed under the Apache 2.0 / MIT License.
