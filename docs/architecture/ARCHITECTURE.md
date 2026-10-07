# DROUGHT-X Architecture & System Design

## 🌵 Project Overview
**DROUGHT-X** is an end-to-end spatial-temporal machine learning framework for multi-horizon drought severity forecasting across all 254 counties in Texas.

---

## 🔄 End-to-End Execution Flow

```text
Streamlit Frontend UI (app.py)
    │
    ▼ HTTP GET /predict/{fips}
FastAPI Inference Backend (backend/main.py)
    │
    ▼
Prediction Pipeline (backend/prediction_pipeline.py)
    ├── 1. Current Drought Monitor Score (backend/usdm.py)
    ├── 2. Spatial Adjacency & Neighbor Lag (backend/neighbours.py)
    ├── 3. Meteorology & Rolling Aggregations (backend/weather.py)
    └── 4. Soil Quality Characteristics (data/soil_data.csv)
    │
    ▼ Feature Assembly (24 engineered features)
Random Forest Multi-Horizon Models (models/)
    ├── model_1w.pkl (t + 1 week horizon)
    ├── model_2w.pkl (t + 2 week horizon)
    ├── model_3w.pkl (t + 3 week horizon)
    └── model_4w.pkl (t + 4 week horizon)
    │
    ▼ Multi-Horizon Forecast Response
Streamlit Interactive Dashboard & Map Visualizations (outputs/maps/)
```

---

## 📂 Directory Layout

| Directory | Purpose |
| :--- | :--- |
| `app.py` | Interactive Streamlit frontend dashboard |
| `backend/` | Production FastAPI backend microservices & inference modules |
| `models/` | Production serialized Random Forest models (`model_1w.pkl` to `model_4w.pkl`) |
| `models/experiments/` | Older and experimental model weights (XGBoost, early RF) |
| `data/` | Production geospatial metadata, centroids, adjacency graph & active caches |
| `data/raw/` | Original large multi-year time-series and benchmark CSV/parquet datasets |
| `data/processed/` | Engineered weekly aggregations, spatial lag tables & clean drought datasets |
| `notebooks/training/` | End-to-end data preparation (`prep.ipynb`) and feature engineering (`preprocess.ipynb`) |
| `notebooks/analysis/` | Validation and visualization workflows (`vis_val.ipynb`) |
| `notebooks/experiments/` | Exploratory modeling, baselines, and meteorological studies |
| `outputs/` | Generated interactive maps, figures, charts, and prediction exports |
| `docs/` | System architecture, reports, and documentation assets |
| `tests/` | Automated system integrity and regression tests |
| `_ARCHIVE/` | Deprecated test stubs, duplicated nested extractions, and archived experiments |
| `_UNRELATED_REVIEW/` | Non-project files (e.g., external reports, practice notebooks) for user review |
