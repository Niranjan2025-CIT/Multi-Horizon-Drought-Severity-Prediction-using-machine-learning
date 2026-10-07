import os
import pandas as pd


# ============================================================
# HISTORICAL DATA
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_weekly.csv"
)
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(
        BASE_DIR,
        "processed_weekly.csv"
    )


# ============================================================
# LOAD DATA ONCE
# ============================================================

historical_df = pd.read_csv(
    DATA_PATH
)

historical_df["fips"] = (
    historical_df["fips"]
    .astype(str)
    .str.zfill(5)
)

historical_df["date"] = pd.to_datetime(
    historical_df["date"],
    errors="coerce"
)

historical_df = (
    historical_df
    .sort_values(
        ["fips", "date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# COUNTY HISTORY
# ============================================================

def get_county_history(fips):

    fips = str(
        fips
    ).zfill(5)

    df = historical_df[
        historical_df["fips"] == fips
    ].copy()

    if df.empty:

        raise ValueError(
            f"No historical data found for FIPS {fips}"
        )

    # --------------------------------------------------------
    # Only send what frontend needs
    # --------------------------------------------------------

    columns = [
        "date",
        "score",
        "PRECTOT",
        "T2M",
        "QV2M",
        "WS10M",
        "target_1w",
        "target_2w",
        "target_3w",
        "target_4w",
        "neighbor_score"
    ]

    columns = [
        col
        for col in columns
        if col in df.columns
    ]

    result = df[columns].copy()

    result["date"] = (
        result["date"]
        .dt.strftime("%Y-%m-%d")
    )
    result = result.astype(object)
   
    result = result.where(
    pd.notna(result),
    None
    )
    return result.to_dict(
        orient="records"
    )


# ============================================================
# LATEST COUNTY DATA
# ============================================================

def get_latest_county_data(fips):

    fips = str(
        fips
    ).zfill(5)

    df = historical_df[
        historical_df["fips"] == fips
    ]

    if df.empty:

        raise ValueError(
            f"No historical data found for {fips}"
        )

    latest = (
        df
        .sort_values("date")
        .iloc[-1]
    )

    return {
        "fips": fips,
        "date": latest["date"].strftime(
            "%Y-%m-%d"
        ),
        "score": float(
            latest["score"]
        ),
        "neighbor_score": float(
            latest["neighbor_score"]
        )
        if pd.notna(
            latest["neighbor_score"]
        )
        else None
    }


# ============================================================
# LATEST DATA FOR ALL COUNTIES
# ============================================================

def get_latest_all_counties():

    latest = (
        historical_df
        .sort_values("date")
        .groupby(
            "fips",
            as_index=False
        )
        .tail(1)
    )

    result = []

    for _, row in latest.iterrows():

        result.append({

            "fips":
                str(row["fips"]).zfill(5),

            "date":
                row["date"].strftime(
                    "%Y-%m-%d"
                ),

            "score":
                float(row["score"])
                if pd.notna(row["score"])
                else None,

            "neighbor_score":
                float(
                    row["neighbor_score"]
                )
                if pd.notna(
                    row["neighbor_score"]
                )
                else None
        })

    return result