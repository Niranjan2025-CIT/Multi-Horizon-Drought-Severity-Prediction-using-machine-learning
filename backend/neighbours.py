# ============================================================
# backend/neighbours.py
# ============================================================

import os
import json
import pandas as pd
import numpy as np

from backend.weather import (
    get_weather_data,
    calculate_weather_features
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
# LOAD NEIGHBOURS
# ============================================================

NEIGHBOUR_FILE = os.path.join(
    BASE_DIR,
    "data",
    "county_neighbors.json"
)


with open(
    NEIGHBOUR_FILE,
    "r",
    encoding="utf-8"
) as f:

    neighbors = json.load(f)


# Normalize every FIPS

neighbors = {
    str(fips).zfill(5): [
        str(n).zfill(5)
        for n in neighbour_list
    ]
    for fips, neighbour_list
    in neighbors.items()
}


# ============================================================
# FEATURES WE NEED FROM NEIGHBOURS
# ============================================================

NEIGHBOR_WEATHER_FEATURES = [
    "PRECTOT_mean_7d",
    "PRECTOT_mean_30d",
    "T2M_mean_30d",
    "QV2M_mean_30d",
    "WS10M_mean_30d"
]


# ============================================================
# GET NEIGHBOUR WEATHER FEATURES
# ============================================================

def get_neighbor_weather_features(fips):

    fips = str(fips).zfill(5)

    neighbour_list = neighbors.get(
        fips,
        []
    )

    if len(neighbour_list) == 0:

        raise ValueError(
            f"No neighbours found for {fips}"
        )

    all_features = []

    print("=" * 60)
    print(f"COUNTY: {fips}")
    print(
        f"NEIGHBOURS: {len(neighbour_list)}"
    )
    print("=" * 60)

    for i, neighbour_fips in enumerate(
        neighbour_list,
        start=1
    ):

        neighbour_fips = (
            str(neighbour_fips)
            .zfill(5)
        )

        print(
            f"\n[{i}/{len(neighbour_list)}] "
            f"{neighbour_fips}"
        )

        try:

            raw_weather = (
                get_weather_data(
                    neighbour_fips
                )
            )

            if not isinstance(
                raw_weather,
                pd.DataFrame
            ):

                raise TypeError(
                    "get_weather_data() "
                    "did not return a DataFrame."
                )

            # PRECTOT should already exist
            # because weather.py renames it.

            if "PRECTOT" not in (
                raw_weather.columns
            ):

                raise ValueError(
                    "Weather data does not "
                    "contain PRECTOT."
                )

            features = (
                calculate_weather_features(
                    raw_weather
                )
            )

            valid = (
                features
                .dropna(
                    subset=(
                        NEIGHBOR_WEATHER_FEATURES
                    )
                )
            )

            if valid.empty:

                print(
                    "   No valid 30-day "
                    "weather features."
                )

                continue

            latest = (
                valid
                .sort_values("date")
                .iloc[-1]
            )

            row = {}

            for feature in (
                NEIGHBOR_WEATHER_FEATURES
            ):

                row[feature] = float(
                    latest[feature]
                )

            all_features.append(row)

            print(
                "   Latest valid date:",
                latest["date"]
            )

            print(
                "   Features collected."
            )

        except Exception as e:

            print(
                "   FAILED:",
                repr(e)
            )

    if not all_features:

        raise ValueError(
            "Could not obtain valid weather "
            "features for any neighbour."
        )

    neighbour_weather_df = pd.DataFrame(
        all_features
    )

    neighbour_mean = (
        neighbour_weather_df
        .mean(
            numeric_only=True
        )
    )

    return neighbour_mean