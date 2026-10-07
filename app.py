
# ============================================================
# app.py
# Texas Drought Prediction Dashboard
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Texas Drought Prediction",
    page_icon="🌵",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

DATA_FILE = (
    os.path.join("data", "processed", "processed_weekly.csv")
    if os.path.exists(os.path.join("data", "processed", "processed_weekly.csv"))
    else "processed_weekly.csv"
)

GEOJSON_FILE = "data/texas_geojson.json"

BACKEND_URL = "http://127.0.0.1:8000"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    df["fips"] = (
        df["fips"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.zfill(5)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_geojson():

    with open(
        GEOJSON_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


df = load_data()

texas_geojson = load_geojson()


# ============================================================
# BASIC DATA CHECK
# ============================================================

if df.empty:

    st.error(
        "processed_weekly.csv is empty."
    )

    st.stop()


# ============================================================
# GET AVAILABLE DATES
# ============================================================

available_dates = (
    df["date"]
    .dropna()
    .sort_values()
    .unique()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌵 Texas Drought Dashboard"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Texas Map",
        "County Details"
    ]
)


# ============================================================
# PAGE 1 — TEXAS MAP
# ============================================================

if page == "Texas Map":

    st.title(
        "🌵 Texas Drought Map"
    )

    st.write(
        "Historical drought severity and "
        "future drought predictions."
    )

    # --------------------------------------------------------
    # DATE SELECTION
    # --------------------------------------------------------

    selected_date = st.select_slider(
        "Select Historical Date",
        options=list(available_dates),
        value=available_dates[-1]
    )

    selected_date = pd.Timestamp(
        selected_date
    )


    # --------------------------------------------------------
    # MAP MODE
    # --------------------------------------------------------

    display_type = st.radio(
        "Map Display",
        [
            "Historical Drought",
            "1 Week Prediction",
            "2 Week Prediction",
            "3 Week Prediction",
            "4 Week Prediction"
        ],
        horizontal=True
    )


    # ========================================================
    # HISTORICAL MAP
    # ========================================================

    if display_type == "Historical Drought":

        # ----------------------------------------------------
        # Get latest row for each county on/before date
        # ----------------------------------------------------

        historical = df[
            df["date"] <= selected_date
        ].copy()

        if historical.empty:

            st.warning(
                "No historical data available."
            )

            st.stop()


        historical = (
            historical
            .sort_values("date")
            .groupby("fips")
            .tail(1)
        )


        # ----------------------------------------------------
        # Build GEO_ID
        # ----------------------------------------------------

        historical["GEO_ID"] = (
            "0500000US"
            + historical["fips"]
        )


        map_data = historical[
            [
                "fips",
                "score",
                "GEO_ID"
            ]
        ].copy()


        # ----------------------------------------------------
        # Make sure score is numeric
        # ----------------------------------------------------

        map_data["score"] = pd.to_numeric(
            map_data["score"],
            errors="coerce"
        )


        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        fig = px.choropleth_mapbox(

            map_data,

            geojson=texas_geojson,

            locations="GEO_ID",

            featureidkey="properties.GEO_ID",

            color="score",

            color_continuous_scale=[
                [0.00, "#2ecc71"],
                [0.20, "#f1c40f"],
                [0.40, "#f39c12"],
                [0.60, "#e74c3c"],
                [0.80, "#8e44ad"],
                [1.00, "#4a235a"]
            ],

            range_color=[0, 5],

            hover_name="fips",

            hover_data={
                "score": ":.2f",
                "GEO_ID": False
            },

            mapbox_style="carto-positron",

            center={
                "lat": 31.0,
                "lon": -99.9
            },

            zoom=5.2,

            opacity=0.8,

            height=750
        )


        fig.update_traces(

            marker_line_width=0.7,

            marker_line_color="white"
        )


        fig.update_layout(

            title={
                "text":
                    "Texas Historical Drought Severity — "
                    + selected_date.strftime(
                        "%d %B %Y"
                    ),

                "x": 0.5
            },

            margin={
                "r": 0,
                "t": 60,
                "l": 0,
                "b": 0
            },

            coloraxis_colorbar={

                "title":
                    "Drought Score",

                "tickvals": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5
                ],

                "ticktext": [

                    "0 — None",

                    "1 — D0",

                    "2 — D1",

                    "3 — D2",

                    "4 — D3",

                    "5 — D4"
                ]
            }
        )


        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "scrollZoom": True,
                "displayModeBar": True
            }
        )


    # ========================================================
    # PREDICTION MAP
    # ========================================================

    else:

        st.subheader(
            "🔮 Future Drought Prediction"
        )

        # ----------------------------------------------------
        # Determine prediction week
        # ----------------------------------------------------

        week = int(
            display_type[0]
        )

        prediction_column = (
            f"pred_{week}w"
        )


        st.info(
            f"Showing drought prediction "
            f"{week} week(s) ahead."
        )


        # ----------------------------------------------------
        # Get latest historical data
        #
        # This gives us all 254 Texas counties
        # ----------------------------------------------------

        latest_data = (
            df
            .sort_values("date")
            .groupby("fips")
            .tail(1)
            .copy()
        )


        # ----------------------------------------------------
        # GEO_ID
        # ----------------------------------------------------

        latest_data["GEO_ID"] = (
            "0500000US"
            + latest_data["fips"]
        )


        # ----------------------------------------------------
        # Create prediction column
        # ----------------------------------------------------

        latest_data[
            prediction_column
        ] = np.nan


        # ----------------------------------------------------
        # Call backend for every county
        # ----------------------------------------------------

        progress = st.progress(
            0
        )

        status = st.empty()

        total = len(
            latest_data
        )

        successful = 0


        for i, fips in enumerate(
            latest_data["fips"]
        ):

            try:

                response = requests.get(

                    f"{BACKEND_URL}/predict/{fips}",

                    timeout=120
                )


                if response.status_code == 200:

                    result = response.json()


                    prediction = result.get(
                        prediction_column
                    )


                    if prediction is not None:

                        latest_data.loc[
                            latest_data["fips"] == fips,
                            prediction_column
                        ] = prediction

                        successful += 1


                status.write(
                    f"Predicting county "
                    f"{i + 1}/{total}: "
                    f"{fips}"
                )


            except Exception as e:

                print(
                    f"Prediction failed "
                    f"for {fips}: {e}"
                )


            progress.progress(
                (i + 1) / total
            )


        status.write(
            f"Prediction complete: "
            f"{successful}/{total} counties"
        )


        # ----------------------------------------------------
        # Check predictions
        # ----------------------------------------------------

        if successful == 0:

            st.error(
                "No predictions were returned "
                "by the backend."
            )

            st.stop()


        # ----------------------------------------------------
        # MAP DATA
        # ----------------------------------------------------

        map_data = latest_data[
            [
                "fips",
                "GEO_ID",
                prediction_column
            ]
        ].copy()


        map_data[
            prediction_column
        ] = pd.to_numeric(

            map_data[
                prediction_column
            ],

            errors="coerce"
        )


        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        fig = px.choropleth_mapbox(

            map_data,

            geojson=texas_geojson,

            locations="GEO_ID",

            featureidkey="properties.GEO_ID",

            color=prediction_column,

            color_continuous_scale=[

                [0.00, "#2ecc71"],

                [0.20, "#f1c40f"],

                [0.40, "#f39c12"],

                [0.60, "#e74c3c"],

                [0.80, "#8e44ad"],

                [1.00, "#4a235a"]
            ],

            range_color=[0, 5],

            hover_name="fips",

            hover_data={
                prediction_column: ":.0f",
                "GEO_ID": False
            },

            mapbox_style="carto-positron",

            center={
                "lat": 31.0,
                "lon": -99.9
            },

            zoom=5.2,

            opacity=0.8,

            height=750
        )


        # ----------------------------------------------------
        # COUNTY BORDERS
        # ----------------------------------------------------

        fig.update_traces(

            marker_line_width=0.7,

            marker_line_color="white"
        )


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        fig.update_layout(

            title={
                "text":
                    f"Texas Drought Prediction — "
                    f"{week} Week Ahead",

                "x": 0.5
            },

            margin={
                "r": 0,
                "t": 60,
                "l": 0,
                "b": 0
            },

            coloraxis_colorbar={

                "title":
                    "Predicted Drought",

                "tickvals": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5
                ],

                "ticktext": [

                    "0 — None",

                    "1 — D0",

                    "2 — D1",

                    "3 — D2",

                    "4 — D3",

                    "5 — D4"
                ]
            }
        )


        # ----------------------------------------------------
        # DISPLAY MAP
        # ----------------------------------------------------

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "scrollZoom": True,
                "displayModeBar": True
            }
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Prediction Summary"
        )


        prediction_counts = (
            map_data[
                prediction_column
            ]
            .value_counts()
            .sort_index()
        )


        for drought_class, count in (
            prediction_counts.items()
        ):

            st.write(
                f"**Drought level "
                f"{int(drought_class)}:** "
                f"{int(count)} counties"
            )


        # ----------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------

        with st.expander(
            "View county predictions"
        ):

            table = map_data.copy()

            table = table.rename(

                columns={
                    prediction_column:
                        "Predicted Drought"
                }
            )

            st.dataframe(
                table[
                    [
                        "fips",
                        "Predicted Drought"
                    ]
                ],

                use_container_width=True
            )


# ============================================================
# PAGE 2 — COUNTY DETAILS
# ============================================================

elif page == "County Details":

    st.title(
        "📍 County Drought Details"
    )


    # --------------------------------------------------------
    # COUNTY SELECTOR
    # --------------------------------------------------------

    county_list = (
        sorted(
            df["fips"]
            .dropna()
            .astype(str)
            .str.zfill(5)
            .unique()
        )
    )


    selected_fips = st.selectbox(

        "Select Texas County",

        county_list,

        index=county_list.index("48001")
        if "48001" in county_list
        else 0
    )


    # --------------------------------------------------------
    # COUNTY HISTORY
    # --------------------------------------------------------

    county_df = (
        df[
            df["fips"]
            .astype(str)
            .str.zfill(5)
            == selected_fips
        ]
        .sort_values("date")
        .copy()
    )


    if county_df.empty:

        st.error(
            "No data found for this county."
        )

        st.stop()


    # --------------------------------------------------------
    # LATEST SCORE
    # --------------------------------------------------------

    latest = (
        county_df
        .iloc[-1]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Drought Score",
            f"{latest['score']:.2f}"
        )


    with col2:

        st.metric(
            "Latest Temperature",
            f"{latest['T2M']:.2f} °C"
        )


    with col3:

        st.metric(
            "Latest Rainfall",
            f"{latest['PRECTOT']:.2f}"
        )


    # ========================================================
    # HISTORICAL DROUGHT GRAPH
    # ========================================================

    st.subheader(
        "Historical Drought Severity"
    )


    fig_history = px.line(

        county_df,

        x="date",

        y="score",

        title=
            f"Historical Drought Score — "
            f"{selected_fips}",

        labels={
            "date": "Date",
            "score": "Drought Score"
        }
    )


    fig_history.update_yaxes(
        range=[0, 5]
    )


    st.plotly_chart(

        fig_history,

        use_container_width=True
    )


    # ========================================================
    # WEATHER GRAPH
    # ========================================================

    st.subheader(
        "Weather History"
    )


    weather_metric = st.selectbox(

        "Select Weather Variable",

        [
            "PRECTOT",
            "T2M",
            "QV2M",
            "WS10M"
        ]
    )


    fig_weather = px.line(

        county_df,

        x="date",

        y=weather_metric,

        title=
            f"{weather_metric} History — "
            f"{selected_fips}"
    )


    st.plotly_chart(

        fig_weather,

        use_container_width=True
    )


    # ========================================================
    # FUTURE PREDICTION
    # ========================================================

    st.subheader(
        "🔮 Future Drought Prediction"
    )


    if st.button(
        "Predict Next 4 Weeks"
    ):

        with st.spinner(
            "Running prediction..."
        ):

            try:

                response = requests.get(

                    f"{BACKEND_URL}/predict/{selected_fips}",

                    timeout=120
                )


                if response.status_code != 200:

                    st.error(
                        f"Backend error: "
                        f"{response.text}"
                    )

                else:

                    prediction = (
                        response.json()
                    )


                    p1 = prediction.get(
                        "pred_1w"
                    )

                    p2 = prediction.get(
                        "pred_2w"
                    )

                    p3 = prediction.get(
                        "pred_3w"
                    )

                    p4 = prediction.get(
                        "pred_4w"
                    )


                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )


                    with col1:

                        st.metric(
                            "1 Week",
                            p1
                        )


                    with col2:

                        st.metric(
                            "2 Weeks",
                            p2
                        )


                    with col3:

                        st.metric(
                            "3 Weeks",
                            p3
                        )


                    with col4:

                        st.metric(
                            "4 Weeks",
                            p4
                        )


                    # ----------------------------------------
                    # Prediction graph
                    # ----------------------------------------

                    prediction_df = pd.DataFrame({

                        "Week": [
                            "Current",
                            "1 Week",
                            "2 Weeks",
                            "3 Weeks",
                            "4 Weeks"
                        ],

                        "Drought Score": [

                            latest["score"],

                            p1,

                            p2,

                            p3,

                            p4
                        ]
                    })


                    fig_prediction = px.line(

                        prediction_df,

                        x="Week",

                        y="Drought Score",

                        markers=True,

                        title=
                            f"Drought Forecast — "
                            f"{selected_fips}"
                    )


                    fig_prediction.update_yaxes(
                        range=[0, 5]
                    )


                    st.plotly_chart(

                        fig_prediction,

                        use_container_width=True
                    )


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


    # ========================================================
    # RAW COUNTY DATA
    # ========================================================

    with st.expander(
        "View historical data"
    ):

        st.dataframe(

            county_df,

            use_container_width=True
        )

