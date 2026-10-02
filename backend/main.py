# ============================================================
# backend/main.py
# ===========================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.prediction_pipeline import (
    predict_county
)

from backend.historical import (
    get_county_history,
    get_latest_county_data,
    get_latest_all_counties
)
from backend.historical import (
    get_county_history,
    get_latest_county_data,
    get_latest_all_counties
)
# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Texas Drought Prediction API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Texas Drought Prediction API"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.get("/predict/{fips}")
def predict(fips: str):

    fips = str(fips).zfill(5)

    try:

        result = predict_county(
            fips
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/history/{fips}")
def history(fips: str):

    try:

        fips = str(fips).zfill(5)

        data = get_county_history(
            fips
        )

        return {
            "fips": fips,
            "data": data
        }

    except Exception as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
@app.get("/latest/{fips}")
def latest(fips: str):

    try:

        fips = str(fips).zfill(5)

        return get_latest_county_data(
            fips
        )

    except Exception as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
@app.get("/map/history")
def map_history():

    try:

        return {
            "data":
                get_latest_all_counties()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

