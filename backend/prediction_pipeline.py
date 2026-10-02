# ============================================================
# backend/prediction_pipeline.py
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PREDICTION_CACHE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "prediction_cache.csv"
)


from backend.usdm import (
    get_usdm_score_for_fips
)

from backend.weather import (
    get_weather_data,
    calculate_weather_features
)

from backend.neighbours import (
    neighbors,
    get_neighbor_weather_features
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# SOIL DATA
# ============================================================

SOIL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "soil_data.csv"
)

soil_features = pd.read_csv(
    SOIL_FILE
)

soil_features["fips"] = (
    soil_features["fips"]
    .astype(str)
    .str.zfill(5)
)


# ============================================================
# EXACT MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    "score",

    "neighbor_score",

    "PRECTOT_mean_7d",

    "PRECTOT_mean_30d",

    "neighbor_PRECTOT_mean_7d",

    "neighbor_PRECTOT_mean_30d",

    "T2M_mean_30d",

    "neighbor_T2M_mean_30d",

    "QV2M_mean_30d",

    "neighbor_QV2M_mean_30d",

    "WS10M_mean_30d",

    "neighbor_WS10M_mean_30d",

    "month_sin",

    "month_cos",

    "day_sin",

    "day_cos",

    "CULT_LAND",

    "SQ1",

    "SQ2",

    "SQ3",

    "SQ4",

    "SQ5",

    "SQ6",

    "SQ7"
]


# ============================================================
# BUILD PREDICTION FEATURES
# ============================================================

def build_prediction_features(fips):

    fips = str(fips).zfill(5)

    print("=" * 70)
    print(
        f"BUILDING FEATURES FOR {fips}"
    )
    print("=" * 70)

    # ========================================================
    # 1. CURRENT USDM SCORE
    # ========================================================

    current_result = (
        get_usdm_score_for_fips(
            fips
        )
    )

    if current_result is None:
        raise ValueError(
            f"No USDM result for {fips}"
        )

    current_score = float(
        current_result["score"]
    )

    print(
        "\nCurrent drought score:",
        current_score
    )

    # ========================================================
    # 2. NEIGHBOUR SCORE
    # ========================================================

    neighbour_list = neighbors.get(
        fips,
        []
    )

    if not neighbour_list:

        raise ValueError(
            f"No neighbours found for {fips}"
        )

    neighbour_scores = []

    print("\nNeighbour scores:")

    for neighbour_fips in neighbour_list:

        neighbour_fips = (
            str(neighbour_fips)
            .zfill(5)
        )

        try:

            result = (
                get_usdm_score_for_fips(
                    neighbour_fips
                )
            )

            if result is None:
                continue

            score = float(
                result["score"]
            )

            if not np.isnan(score):

                neighbour_scores.append(
                    score
                )

                print(
                    f"{neighbour_fips}: "
                    f"{score}"
                )

        except Exception as e:

            print(
                f"{neighbour_fips}: "
                f"ERROR - {e}"
            )

    if not neighbour_scores:

        raise ValueError(
            "No valid neighbour scores."
        )

    neighbour_score = float(
        np.mean(
            neighbour_scores
        )
    )

    print(
        "\nNeighbour average:",
        neighbour_score
    )

    # ========================================================
    # 3. CURRENT WEATHER
    # ========================================================

    current_weather = (
        get_weather_data(
            fips
        )
    )

    current_weather_features = (
        calculate_weather_features(
            current_weather
        )
    )

    required_weather = [
        "PRECTOT_mean_7d",
        "PRECTOT_mean_30d",
        "T2M_mean_30d",
        "QV2M_mean_30d",
        "WS10M_mean_30d"
    ]

    valid_current = (
        current_weather_features
        .dropna(
            subset=required_weather
        )
    )

    if valid_current.empty:

        raise ValueError(
            "No valid current weather "
            "features."
        )

    latest_current = (
        valid_current
        .sort_values("date")
        .iloc[-1]
    )

    latest_date = pd.to_datetime(
        latest_current["date"]
    )

    print(
        "\nLatest weather date:",
        latest_date
    )

    # ========================================================
    # 4. NEIGHBOUR WEATHER
    # ========================================================

    neighbour_weather = (
        get_neighbor_weather_features(
            fips
        )
    )

    # ========================================================
    # 5. CYCLICAL DATE FEATURES
    # ========================================================

    month = latest_date.month

    day_of_year = (
        latest_date.dayofyear
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    day_sin = np.sin(
        2 * np.pi * day_of_year / 365
    )

    day_cos = np.cos(
        2 * np.pi * day_of_year / 365
    )

    # ========================================================
    # 6. SOIL DATA
    # ========================================================

    soil_row = soil_features[
        soil_features["fips"] == fips
    ]

    if soil_row.empty:

        raise ValueError(
            f"No soil data found for {fips}"
        )

    soil_row = soil_row.iloc[0]

    # ========================================================
    # 7. BUILD FEATURE DICTIONARY
    # ========================================================

    feature_dict = {

        "score":
            current_score,

        "neighbor_score":
            neighbour_score,

        "PRECTOT_mean_7d":
            float(
                latest_current[
                    "PRECTOT_mean_7d"
                ]
            ),

        "PRECTOT_mean_30d":
            float(
                latest_current[
                    "PRECTOT_mean_30d"
                ]
            ),

        "neighbor_PRECTOT_mean_7d":
            float(
                neighbour_weather[
                    "PRECTOT_mean_7d"
                ]
            ),

        "neighbor_PRECTOT_mean_30d":
            float(
                neighbour_weather[
                    "PRECTOT_mean_30d"
                ]
            ),

        "T2M_mean_30d":
            float(
                latest_current[
                    "T2M_mean_30d"
                ]
            ),

        "neighbor_T2M_mean_30d":
            float(
                neighbour_weather[
                    "T2M_mean_30d"
                ]
            ),

        "QV2M_mean_30d":
            float(
                latest_current[
                    "QV2M_mean_30d"
                ]
            ),

        "neighbor_QV2M_mean_30d":
            float(
                neighbour_weather[
                    "QV2M_mean_30d"
                ]
            ),

        "WS10M_mean_30d":
            float(
                latest_current[
                    "WS10M_mean_30d"
                ]
            ),

        "neighbor_WS10M_mean_30d":
            float(
                neighbour_weather[
                    "WS10M_mean_30d"
                ]
            ),

        "month_sin":
            float(month_sin),

        "month_cos":
            float(month_cos),

        "day_sin":
            float(day_sin),

        "day_cos":
            float(day_cos)
    }

    # ========================================================
    # 8. SOIL FEATURES
    # ========================================================

    soil_columns = [
        "CULT_LAND",
        "SQ1",
        "SQ2",
        "SQ3",
        "SQ4",
        "SQ5",
        "SQ6",
        "SQ7"
    ]

    for col in soil_columns:

        feature_dict[col] = float(
            soil_row[col]
        )

    # ========================================================
    # 9. CREATE DATAFRAME
    # ========================================================

    X_live = pd.DataFrame(
        [feature_dict]
    )

    # ========================================================
    # 10. EXACT FEATURE ORDER
    # ========================================================

    X_live = X_live[
        MODEL_FEATURES
    ]

    # ========================================================
    # 11. CHECK
    # ========================================================

    if list(X_live.columns) != MODEL_FEATURES:

        raise ValueError(
            "Feature order mismatch."
        )

    if X_live.shape != (1, 24):

        raise ValueError(
            f"Expected (1, 24), "
            f"got {X_live.shape}"
        )

    print(
        "\nFeature vector shape:",
        X_live.shape
    )

    return X_live

# ============================================================
# LOAD PREDICTION CACHE
# ============================================================

def load_prediction_cache():

    if not os.path.exists(
        PREDICTION_CACHE_FILE
    ):
        return pd.DataFrame(
            columns=[
                "fips",
                "prediction_date",
                "pred_1w",
                "pred_2w",
                "pred_3w",
                "pred_4w"
            ]
        )

    cache = pd.read_csv(
        PREDICTION_CACHE_FILE
    )

    if not cache.empty:

        cache["fips"] = (
            cache["fips"]
            .astype(str)
            .str.zfill(5)
        )

        cache["prediction_date"] = (
            pd.to_datetime(
                cache["prediction_date"],
                errors="coerce"
            )
        )

    return cache


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    fips,
    prediction_date,
    predictions
):

    fips = str(fips).zfill(5)

    cache = load_prediction_cache()

    new_row = pd.DataFrame([
        {
            "fips": fips,
            "prediction_date": prediction_date,

            "pred_1w":
                predictions["pred_1w"],

            "pred_2w":
                predictions["pred_2w"],

            "pred_3w":
                predictions["pred_3w"],

            "pred_4w":
                predictions["pred_4w"]
        }
    ])

    # --------------------------------------------------------
    # Remove an old prediction for the same
    # county + prediction date
    # --------------------------------------------------------

    if not cache.empty:

        cache = cache[
            ~(
                (cache["fips"] == fips)
                &
                (
                    cache["prediction_date"]
                    == pd.to_datetime(
                        prediction_date
                    )
                )
            )
        ]

    cache = pd.concat(
        [
            cache,
            new_row
        ],
        ignore_index=True
    )

    cache.to_csv(
        PREDICTION_CACHE_FILE,
        index=False
    )

    print(
        f"Prediction cache saved: "
        f"{fips} | {prediction_date}"
    )


# ============================================================
# GET CACHED PREDICTION
# ============================================================

def get_cached_prediction(
    fips,
    prediction_date
):

    fips = str(fips).zfill(5)

    cache = load_prediction_cache()

    if cache.empty:
        return None

    prediction_date = pd.to_datetime(
        prediction_date
    )

    match = cache[
        (cache["fips"] == fips)
        &
        (
            cache["prediction_date"]
            == prediction_date
        )
    ]

    if match.empty:
        return None

    row = match.iloc[0]

    print(
        f"PREDICTION CACHE HIT — "
        f"{fips} | {prediction_date.date()}"
    )

    return {
        "fips": fips,

        "pred_1w":
            int(row["pred_1w"]),

        "pred_2w":
            int(row["pred_2w"]),

        "pred_3w":
            int(row["pred_3w"]),

        "pred_4w":
            int(row["pred_4w"])
    }
# ============================================================
# PREDICT COUNTY
# ============================================================

def predict_county(fips):

    fips = str(fips).zfill(5)

    print("=" * 70)
    print(
        f"PREDICTING DROUGHT FOR COUNTY: {fips}"
    )
    print("=" * 70)

    # ========================================================
    # FIND LATEST WEATHER DATE
    # ========================================================

    weather = get_weather_data(fips)

    if weather.empty:

        raise ValueError(
            f"No weather data available for {fips}"
        )

    prediction_date = pd.to_datetime(
        weather["date"]
    ).max()

    print(
        f"Prediction date: "
        f"{prediction_date.date()}"
    )

    # ========================================================
    # CHECK PREDICTION CACHE
    # ========================================================

    cached = get_cached_prediction(
        fips,
        prediction_date
    )

    if cached is not None:

        print(
            "Using cached prediction."
        )

        return cached

    # ========================================================
    # CACHE MISS
    # ========================================================

    print(
        "PREDICTION CACHE MISS"
    )

    # ========================================================
    # BUILD 24 FEATURES
    # ========================================================

    X_live = build_prediction_features(
        fips
    )

    # ========================================================
    # LOAD FOUR MODELS
    # ========================================================

    predictions = {}

    for week in range(1, 5):

        model_path = os.path.join(
            BASE_DIR,
            "models",
            f"model_{week}w.pkl"
        )

        print(
            f"Loading model: "
            f"model_{week}w.pkl"
        )

        model = joblib.load(
            model_path
        )

        pred = model.predict(
            X_live
        )[0]

        predictions[
            f"pred_{week}w"
        ] = int(pred)

    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {
        "fips": fips,
        **predictions
    }

    save_prediction(
        fips,
        prediction_date,
        result
    )

    print(
        "Prediction:",
        result
    )

    return result