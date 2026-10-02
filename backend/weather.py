
import os
import requests
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FILE PATHS
# ============================================================

COUNTY_LOCATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "texas_county_locations.csv"
)

WEATHER_CACHE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "weather_cache.csv"
)


# ============================================================
# COUNTY LOCATIONS
# ============================================================

county_locations = pd.read_csv(
    COUNTY_LOCATION_FILE
)

county_locations["fips"] = (
    county_locations["fips"]
    .astype(str)
    .str.zfill(5)
)


# ============================================================
# WEATHER CACHE COLUMNS
# ============================================================

WEATHER_COLUMNS = [
    "fips",
    "date",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "QV2M",
    "WS10M",
    "PRECTOT"
]


# ============================================================
# LOAD WEATHER CACHE
# ============================================================

def load_weather_cache():

    if not os.path.exists(WEATHER_CACHE_FILE):

        return pd.DataFrame(
            columns=WEATHER_COLUMNS
        )

    cache = pd.read_csv(
        WEATHER_CACHE_FILE
    )

    if cache.empty:

        return pd.DataFrame(
            columns=WEATHER_COLUMNS
        )

    cache["fips"] = (
        cache["fips"]
        .astype(str)
        .str.zfill(5)
    )

    cache["date"] = pd.to_datetime(
        cache["date"],
        errors="coerce"
    )

    numeric_columns = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "QV2M",
        "WS10M",
        "PRECTOT"
    ]

    for col in numeric_columns:

        if col in cache.columns:

            cache[col] = pd.to_numeric(
                cache[col],
                errors="coerce"
            )

    return cache


# ============================================================
# SAVE WEATHER CACHE
# ============================================================

def save_weather_cache(cache):

    cache = cache.copy()

    cache["fips"] = (
        cache["fips"]
        .astype(str)
        .str.zfill(5)
    )

    cache["date"] = pd.to_datetime(
        cache["date"],
        errors="coerce"
    )

    cache = (
        cache[
            WEATHER_COLUMNS
        ]
        .drop_duplicates(
            subset=["fips", "date"],
            keep="last"
        )
        .sort_values(
            ["fips", "date"]
        )
        .reset_index(drop=True)
    )

    cache.to_csv(
        WEATHER_CACHE_FILE,
        index=False
    )

    print(
        f"Weather cache saved: "
        f"{len(cache)} rows"
    )


# ============================================================
# FIND COUNTY COORDINATES
# ============================================================

def get_county_coordinates(fips):

    fips = str(fips).zfill(5)

    county_match = county_locations[
        county_locations["fips"] == fips
    ]

    if county_match.empty:

        raise ValueError(
            f"FIPS {fips} not found in "
            "texas_county_locations.csv"
        )

    county = county_match.iloc[0]

    latitude = float(
        county["latitude"]
    )

    longitude = float(
        county["longitude"]
    )

    return latitude, longitude


# ============================================================
# FETCH WEATHER FROM NASA
# ============================================================

def fetch_weather_from_nasa(
    fips,
    start_date,
    end_date
):

    fips = str(fips).zfill(5)

    latitude, longitude = (
        get_county_coordinates(fips)
    )

    print(
        f"NASA API request: "
        f"{fips} | "
        f"{start_date.date()} -> "
        f"{end_date.date()}"
    )

    url = (
        "https://power.larc.nasa.gov/"
        "api/temporal/daily/point"
    )

    params = {

        "parameters": (
            "T2M,"
            "T2M_MAX,"
            "T2M_MIN,"
            "QV2M,"
            "WS10M,"
            "PRECTOTCORR"
        ),

        "community": "AG",

        "longitude": longitude,

        "latitude": latitude,

        "start": start_date.strftime(
            "%Y%m%d"
        ),

        "end": end_date.strftime(
            "%Y%m%d"
        ),

        "format": "JSON",

        "time-standard": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    parameter_data = (
        data["properties"]["parameter"]
    )

    weather = pd.DataFrame(
        parameter_data
    )

    weather.index.name = "date"

    weather = (
        weather
        .reset_index()
    )

    weather["date"] = pd.to_datetime(
        weather["date"],
        format="%Y%m%d",
        errors="coerce"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # NASA PRECTOTCORR
    # becomes PRECTOT
    # --------------------------------------------------------

    weather = weather.rename(
        columns={
            "PRECTOTCORR": "PRECTOT"
        }
    )

    weather["fips"] = fips

    numeric_columns = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "QV2M",
        "WS10M",
        "PRECTOT"
    ]

    for col in numeric_columns:

        weather[col] = pd.to_numeric(
            weather[col],
            errors="coerce"
        )

        # NASA invalid values
        weather.loc[
            weather[col] <= -900,
            col
        ] = np.nan

    weather = weather[
        WEATHER_COLUMNS
    ]

    weather = (
        weather
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"NASA returned "
        f"{len(weather)} rows for {fips}"
    )

    return weather


# ============================================================
# GET WEATHER DATA — CACHE FIRST
# ============================================================

def get_weather_data(
    fips,
    start_date="2026-07-01",
    end_date="2026-08-08"
):

    # --------------------------------------------------------
    # Normalize FIPS
    # --------------------------------------------------------

    fips = str(fips).zfill(5)

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    requested_start = pd.to_datetime(
        start_date
    )

    requested_end = pd.to_datetime(
        end_date
    )

    if requested_start > requested_end:

        raise ValueError(
            "start_date cannot be after "
            "end_date."
        )

    print()
    print("=" * 70)
    print(
        f"WEATHER REQUEST: {fips}"
    )
    print(
        f"Required: "
        f"{requested_start.date()} -> "
        f"{requested_end.date()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------

    cache = load_weather_cache()

    county_cache = cache[
        cache["fips"] == fips
    ].copy()

    county_cache["date"] = pd.to_datetime(
        county_cache["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Determine which dates are missing
    # --------------------------------------------------------

    required_dates = pd.date_range(
        start=requested_start,
        end=requested_end,
        freq="D"
    )

    cached_dates = set(
        county_cache["date"]
        .dropna()
        .dt.normalize()
    )

    missing_dates = [
        date
        for date in required_dates
        if date.normalize()
        not in cached_dates
    ]

    # --------------------------------------------------------
    # EVERYTHING AVAILABLE
    # --------------------------------------------------------

    if not missing_dates:

        print(
            "CACHE HIT — no API request needed."
        )

        result = county_cache[
            county_cache["date"].between(
                requested_start,
                requested_end
            )
        ].copy()

        return (
            result
            .sort_values("date")
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # SOME DATA IS MISSING
    # --------------------------------------------------------

    print(
        f"CACHE MISS — "
        f"{len(missing_dates)} dates missing."
    )

    # --------------------------------------------------------
    # Find continuous missing range(s)
    #
    # Usually this will be one range, e.g.
    # 2026-08-09 -> 2026-08-09
    # --------------------------------------------------------

    missing_dates = sorted(
        missing_dates
    )

    ranges = []

    range_start = missing_dates[0]
    previous = missing_dates[0]

    for current in missing_dates[1:]:

        if (
            current
            == previous + pd.Timedelta(days=1)
        ):

            previous = current

        else:

            ranges.append(
                (
                    range_start,
                    previous
                )
            )

            range_start = current
            previous = current

    ranges.append(
        (
            range_start,
            previous
        )
    )

    # --------------------------------------------------------
    # Fetch only missing ranges
    # --------------------------------------------------------

    new_data = []

    for fetch_start, fetch_end in ranges:

        fetched = fetch_weather_from_nasa(
            fips,
            fetch_start,
            fetch_end
        )

        if not fetched.empty:

            new_data.append(
                fetched
            )

    # --------------------------------------------------------
    # Add new data to cache
    # --------------------------------------------------------

    if new_data:

        new_weather = pd.concat(
            new_data,
            ignore_index=True
        )

        cache = pd.concat(
            [
                cache,
                new_weather
            ],
            ignore_index=True
        )

        save_weather_cache(
            cache
        )

    # --------------------------------------------------------
    # Reload cache
    # --------------------------------------------------------

    cache = load_weather_cache()

    result = cache[
        (cache["fips"] == fips)
        &
        (
            cache["date"].between(
                requested_start,
                requested_end
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["fips", "date"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Returning "
        f"{len(result)} weather rows."
    )

    print(
        f"Latest cached date: "
        f"{result['date'].max()}"
    )

    return result


# ============================================================
# CALCULATE WEATHER FEATURES
# ============================================================

def calculate_weather_features(
    weather_df
):

    df = weather_df.copy()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "T2M",
        "QV2M",
        "WS10M",
        "PRECTOT"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing weather columns: "
            f"{missing}\n"
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Numeric weather columns
    # --------------------------------------------------------

    weather_columns = [
        "T2M",
        "QV2M",
        "WS10M",
        "PRECTOT"
    ]

    for col in weather_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Invalid NASA values
    # --------------------------------------------------------

    df[weather_columns] = (
        df[weather_columns]
        .replace(-999, np.nan)
    )

    # Also handle very negative NASA fill values
    df[weather_columns] = (
        df[weather_columns]
        .mask(
            df[weather_columns] <= -900
        )
    )

    # --------------------------------------------------------
    # 7-day precipitation mean
    # --------------------------------------------------------

    df["PRECTOT_mean_7d"] = (
        df["PRECTOT"]
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    # --------------------------------------------------------
    # 30-day precipitation mean
    # --------------------------------------------------------

    df["PRECTOT_mean_30d"] = (
        df["PRECTOT"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # 30-day temperature mean
    # --------------------------------------------------------

    df["T2M_mean_30d"] = (
        df["T2M"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # 30-day humidity mean
    # --------------------------------------------------------

    df["QV2M_mean_30d"] = (
        df["QV2M"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # 30-day wind-speed mean
    # --------------------------------------------------------

    df["WS10M_mean_30d"] = (
        df["WS10M"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    return df

