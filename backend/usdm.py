import requests
import pandas as pd
import numpy as np
import io


# ============================================================
# USDM API
# ============================================================

USDM_URL = (
    "https://usdmdataservices.unl.edu/"
    "api/CountyStatistics/"
    "GetDroughtSeverityStatisticsByArea"
)


# ============================================================
# Drought columns
# ============================================================

DROUGHT_COLUMNS = [
    "D0",
    "D1",
    "D2",
    "D3",
    "D4"
]


# ============================================================
# Drought class → model score
# ============================================================

def drought_class_to_score(drought_class):

    if pd.isna(drought_class):
        return np.nan

    # Temporary conversion.
    # Our model was trained using 0–5 scores.
    #
    # USDM classes:
    # 0 = None
    # 1 = D0
    # 2 = D1
    # 3 = D2
    # 4 = D3
    # 5 = D4

    return float(drought_class)*5/4


# ============================================================
# Determine drought class from one USDM row
# ============================================================

def usdm_row_to_class(row):

    # Check from most severe → least severe.
    #
    # D4 → class 5
    # D3 → class 4
    # D2 → class 3
    # D1 → class 2
    # D0 → class 1
    # None → class 0

    for drought_column in [
        "D4",
        "D3",
        "D2",
        "D1",
        "D0"
    ]:

        value = row[drought_column]

        value = pd.to_numeric(
            value,
            errors="coerce"
        )

        if pd.notna(value) and value > 0:

            return (
                DROUGHT_COLUMNS.index(
                    drought_column
                ) + 1
            )

    return 0


# ============================================================
# GET USDM SCORE FOR ONE COUNTY
# ============================================================

def get_usdm_score_for_fips(
    fips,
    start_date="01/01/2026",
    end_date="08/08/2026"
):

    # --------------------------------------------------------
    # Standardize FIPS
    # --------------------------------------------------------

    fips = str(
        fips
    ).zfill(5)


    # --------------------------------------------------------
    # API parameters
    # --------------------------------------------------------

    params = {

        "aoi": fips,

        "startdate": start_date,

        "enddate": end_date,

        "statisticsType": "1"
    }


    # --------------------------------------------------------
    # Request USDM
    # --------------------------------------------------------

    response = requests.get(

        USDM_URL,

        params=params,

        timeout=30
    )


    print(
        f"USDM request: {fips} | "
        f"HTTP {response.status_code}"
    )


    response.raise_for_status()


    # --------------------------------------------------------
    # Convert CSV response → DataFrame
    # --------------------------------------------------------

    df = pd.read_csv(
        io.StringIO(
            response.text
        )
    )


    if df.empty:

        return {

            "fips": fips,

            "drought_class": np.nan,

            "score": np.nan,

            "map_date": None
        }


    # --------------------------------------------------------
    # Convert D0-D4 to numeric
    #
    # Important because some columns may arrive as strings.
    # --------------------------------------------------------

    for col in DROUGHT_COLUMNS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        else:

            raise ValueError(
                f"Missing USDM column: {col}"
            )


    # --------------------------------------------------------
    # Convert MapDate
    # --------------------------------------------------------

    df["MapDate"] = pd.to_numeric(
        df["MapDate"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    df = df.dropna(
        subset=["MapDate"]
    )


    if df.empty:

        return {

            "fips": fips,

            "drought_class": np.nan,

            "score": np.nan,

            "map_date": None
        }


    # --------------------------------------------------------
    # Latest USDM week
    # --------------------------------------------------------

    df = df.sort_values(
        "MapDate",
        ascending=False
    )


    latest = df.iloc[0]


    # --------------------------------------------------------
    # Determine drought class
    # --------------------------------------------------------

    drought_class = (
        usdm_row_to_class(
            latest
        )
    )


    # --------------------------------------------------------
    # Convert class → model score
    # --------------------------------------------------------

    drought_score = (
        drought_class_to_score(
            drought_class
        )
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "fips": fips,

        "drought_class":
            drought_class,

        "score":
            drought_score,

        "map_date":
            latest["MapDate"]
    }