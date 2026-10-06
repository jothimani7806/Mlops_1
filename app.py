import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import WarningMessage

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "used_cars.csv"
MODEL_PATH = PROJECT_DIR / "artifacts" / "models" / "best_model.pkl"
METADATA_PATH = PROJECT_DIR / "artifacts" / "models" / "model_metadata.json"
COMPARISON_PATH = PROJECT_DIR / "artifacts" / "reports" / "comparison.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    color: #172033;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #687386;
    margin-bottom: 25px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e7eaf0;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
}

.prediction-card {
    background: linear-gradient(135deg, #172033, #334e7d);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
}

.prediction-price {
    font-size: 42px;
    font-weight: 700;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    color: #172033;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    # Convert price from "$123,456" string to numeric
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convert mileage from "123,456 mi." to numeric
    if "milage" in df.columns:
        df["milage"] = (
            df["milage"]
            .astype(str)
            .str.replace(" mi.", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    return df


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD METADATA
# ============================================================


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return {}

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


# ============================================================
# LOAD FILES
# ============================================================

df = load_data()
model = load_model()
metadata = load_metadata()


# ============================================================
# CHECK FILES
# ============================================================

if df is None:

    st.error("❌ Dataset not found.\n\n" "Expected location:\n" "`data/used_cars.csv`")

    st.stop()


if model is None:

    st.error(
        "❌ Trained model not found.\n\n"
        "Run your Used_Car_Pred_MLOPS.ipynb first "
        "and make sure `artifacts/models/best_model.pkl` exists."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Used Car AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "💰 Price Prediction",
        "🤖 Model Performance",
        "📊 Dataset Explorer",
    ],
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-powered used car price prediction " "using machine learning and MLflow."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="dashboard-title">🚗 Used Car Price Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Machine Learning Dashboard for Used Car Market Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_cars = len(df)

    avg_price = df["price"].mean()

    median_price = df["price"].median()

    max_price = df["price"].max()

    best_model = metadata.get("model_name", "Unknown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric("🚗 Total Cars", f"{total_cars:,}")

    with col2:

        st.metric("💰 Average Price", f"${avg_price:,.0f}")

    with col3:

        st.metric("📊 Median Price", f"${median_price:,.0f}")

    with col4:

        st.metric("🏆 Best Model", best_model)

    st.markdown(
        '<div class="section-title">📈 Market Overview</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        st.subheader("💰 Price Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(df["price"], bins=40, kde=True, ax=ax)

        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Number of Cars")

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # TOP BRANDS
    # --------------------------------------------------------

    with col2:

        st.subheader("🚘 Cars by Brand")

        brand_count = df["brand"].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(x=brand_count.values, y=brand_count.index, ax=ax)

        ax.set_xlabel("Number of Cars")
        ax.set_ylabel("Brand")

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⛽ Price by Fuel Type")

        fuel_price = (
            df.groupby("fuel_type")["price"].median().sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(x=fuel_price.index, y=fuel_price.values, ax=ax)

        ax.set_ylabel("Median Price ($)")
        ax.set_xlabel("Fuel Type")

        plt.xticks(rotation=30)

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("⚠️ Accident Impact")

        if "accident" in df.columns:

            accident_price = df.groupby("accident")["price"].median()

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.barplot(x=accident_price.index, y=accident_price.values, ax=ax)

            ax.set_ylabel("Median Price ($)")
            ax.set_xlabel("Accident Status")

            plt.xticks(rotation=20)

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "💰 Price Prediction":

    st.markdown(
        '<div class="dashboard-title">💰 Car Price Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Enter the car details below and the trained ML model "
        "will estimate its price."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    with col1:

        st.subheader("🚘 Car Information")

        brands = sorted(df["brand"].dropna().unique().tolist())

        brand = st.selectbox("Brand", brands)

        model_name = st.text_input("Model", "Camry")

        model_year = st.number_input(
            "Model Year", min_value=1990, max_value=2026, value=2020, step=1
        )

        mileage = st.number_input(
            "Mileage (miles)", min_value=0, max_value=500000, value=50000, step=1000
        )

        engine = st.text_input("Engine", "2.5L I4 203HP")

    with col2:

        st.subheader("⚙️ Vehicle Details")

        fuel_types = sorted(df["fuel_type"].dropna().astype(str).unique().tolist())

        fuel_type = st.selectbox("Fuel Type", fuel_types)

        transmission_options = sorted(
            df["transmission"].dropna().astype(str).unique().tolist()
        )

        transmission = st.selectbox("Transmission", transmission_options)

        accident_options = sorted(df["accident"].dropna().astype(str).unique().tolist())

        accident = st.selectbox("Accident Status", accident_options)

        clean_title = st.selectbox("Clean Title", ["Yes", "No"])

        int_col = st.text_input("Interior Color", "Black")

        ext_col = st.text_input("Exterior Color", "White")

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button("🚀 Predict Car Price", use_container_width=True):

        try:

            # -----------------------------------------------
            # FEATURE ENGINEERING
            # -----------------------------------------------

            vehicle_age = 2025 - model_year

            if vehicle_age <= 0:

                vehicle_age = 1

            mileage_per_year = mileage / vehicle_age

            # Extract horsepower

            import re

            hp_match = re.search(r"(\d+\.\d+|\d+)\s*HP", engine.upper())

            if hp_match:

                hp = float(hp_match.group(1))

            else:

                hp = float(
                    df["engine"]
                    .str.extract(r"(\d+\.\d+)HP", expand=False)
                    .dropna()
                    .astype(float)
                    .median()
                )

            # Extract engine displacement

            displacement_match = re.search(r"(\d+\.\d+|\d+)\s*L", engine.upper())

            if displacement_match:

                engine_displacement = float(displacement_match.group(1))

            else:

                engine_displacement = float(
                    df["engine displacement"].dropna().median()
                    if "engine displacement" in df.columns
                    else 2.5
                )

            # V engine

            is_v_engine = bool(re.search(r"V\d+", engine.upper()))

            # Accident

            accident_impact = (
                1 if accident == "At least 1 accident or damage reported" else 0
            )

            # Clean title

            clean_title_value = 1 if clean_title == "Yes" else 0

            # -----------------------------------------------
            # BUILD INPUT
            # -----------------------------------------------

            input_data = pd.DataFrame(
                {
                    "brand": [brand],
                    "fuel_type": [fuel_type],
                    "transmission": [transmission],
                    "hp": [hp],
                    "engine displacement": [engine_displacement],
                    "is_v_engine": [is_v_engine],
                    "Accident_Impact": [accident_impact],
                    "clean_title": [clean_title_value],
                    "Vehicle_Age": [vehicle_age],
                    "Mileage_per_Year": [mileage_per_year],
                }
            )

            # -----------------------------------------------
            # ADD AGE / MILEAGE BINS
            # -----------------------------------------------

            try:

                age_edges = metadata["feature_preparation"]["vehicle_age_bin_edges"]

                mileage_edges = metadata["feature_preparation"]["mileage_bin_edges"]

                input_data["Age_Mid"] = int(vehicle_age > age_edges[1])

                input_data["Age_Old"] = int(vehicle_age > age_edges[2])

                input_data["Age_Very Old"] = int(vehicle_age > age_edges[3])

                input_data["Milage_Medium"] = int(mileage > mileage_edges[1])

                input_data["Milage_High"] = int(mileage > mileage_edges[2])

                input_data["Milage_Very High"] = int(mileage > mileage_edges[3])

            except Exception:

                pass

            # -----------------------------------------------
            # ENSURE EXPECTED FEATURES
            # -----------------------------------------------

            expected_features = metadata.get("model_input", {}).get(
                "feature_columns", []
            )

            if expected_features:

                for column in expected_features:

                    if column not in input_data.columns:

                        input_data[column] = 0

                input_data = input_data[expected_features]

            # -----------------------------------------------
            # PREDICTION
            # -----------------------------------------------

            prediction = model.predict(input_data)[0]

            prediction = max(0, float(prediction))

            # -----------------------------------------------
            # DISPLAY
            # -----------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div>🚗 Estimated Used Car Price</div>

                    <div class="prediction-price">
                        ${prediction:,.0f}
                    </div>

                    <div>
                        AI-generated price estimate
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success("✅ Prediction completed successfully!")

        except Exception as e:

            st.error(f"Prediction error: {e}")

            st.info(
                "Your saved model expects the exact "
                "feature structure created during notebook preprocessing."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="dashboard-title">🤖 Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown("Comparison of the machine learning models trained in your notebook.")

    if COMPARISON_PATH.exists():

        comparison = pd.read_csv(COMPARISON_PATH)

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model_name = metadata.get("model_name", comparison.iloc[0]["model"])

        best_row = comparison[comparison["model"] == best_model_name]

        if len(best_row) > 0:

            best_row = best_row.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("🏆 Best Model", best_model_name)

            with col2:

                st.metric("RMSE", f"${best_row['test_rmse']:,.0f}")

            with col3:

                st.metric("R² Score", f"{best_row['test_r2']:.3f}")

        st.markdown("---")

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader("📋 Model Comparison")

        display_columns = ["model", "test_mae", "test_rmse", "test_r2", "test_mape"]

        available_columns = [c for c in display_columns if c in comparison.columns]

        st.dataframe(
            comparison[available_columns], use_container_width=True, hide_index=True
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        st.subheader("📉 Test RMSE — Lower is Better")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(data=comparison, x="model", y="test_rmse", ax=ax)

        ax.set_xlabel("Model")
        ax.set_ylabel("Test RMSE ($)")

        plt.xticks(rotation=25)

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # R2
        # ----------------------------------------------------

        st.subheader("📈 Test R² — Higher is Better")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(data=comparison, x="model", y="test_r2", ax=ax)

        ax.set_xlabel("Model")
        ax.set_ylabel("R² Score")

        plt.xticks(rotation=25)

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning("⚠️ comparison.csv not found.")

        st.info("Run the MLflow training section of " "your notebook first.")


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.markdown(
        '<div class="dashboard-title">📊 Dataset Explorer</div>', unsafe_allow_html=True
    )

    st.markdown("Explore the used car dataset used for training.")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        brands = sorted(df["brand"].dropna().unique().tolist())

        selected_brands = st.multiselect("Select Brands", brands, default=brands[:5])

    with col2:

        max_price_filter = st.slider(
            "Maximum Price",
            int(df["price"].min()),
            int(df["price"].max()),
            int(df["price"].max()),
        )

    filtered_df = df.copy()

    if selected_brands:

        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brands)]

    filtered_df = filtered_df[filtered_df["price"] <= max_price_filter]

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    st.subheader(f"🚗 {len(filtered_df):,} Cars Found")

    st.dataframe(filtered_df.head(500), use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader("📈 Dataset Statistics")

    st.write(filtered_df.describe())
