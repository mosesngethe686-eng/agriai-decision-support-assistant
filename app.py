
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="AgriAI Decision Support Assistant",
    page_icon="🌱",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/mosesngethe686-eng/agriai-decision-support-assistant/main/final_model_dataset.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data.sort_values("year").reset_index(drop=True)

@st.cache_resource
def train_beans_model():
    data = load_data()

    yield_columns = [
        "bean_yield", "maize_yield", "potato_yield"
    ]

    lag_columns = [
        "bean_yield_lag1",
        "maize_yield_lag1",
        "potato_yield_lag1"
    ]

    base_features = [
        col for col in data.columns
        if col not in yield_columns + lag_columns
    ]

    features = base_features + ["bean_yield_lag1"]

    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(train[features], train["bean_yield"])

    return model, features

st.title("🌱 AgriAI Decision Support Assistant")

st.write(
    "An exploratory agricultural decision-support application "
    "for crop-yield estimation in Kenya."
)

st.info(
    "This prototype uses historical FAO agricultural data "
    "from 1962–2023. Predictions are experimental and should "
    "not be treated as guaranteed harvest outcomes."
)

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

crop = st.selectbox(
    "Select the crop",
    ["Beans", "Maize", "Potatoes"]
)

st.divider()

if crop == "Beans":

    st.subheader("Bean yield prediction")

    st.write(
        "Enter the agricultural information below. "
        "Default values are historical medians and should "
        "be replaced with actual information where available."
    )

    model, features = train_beans_model()

    input_values = {}

    with st.form("beans_form"):

        for i in range(0, len(features), 2):
            cols = st.columns(2)

            for j, feature in enumerate(features[i:i+2]):
                default_value = float(df[feature].median())

                with cols[j]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        format="%.4f"
                    )

        submitted = st.form_submit_button(
            "Estimate bean yield"
        )

    if submitted:
        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated bean yield: {prediction:,.2f}"
        )

        st.caption(
            "Prediction generated using the Random Forest model. "
            "This is an experimental estimate, not a guarantee."
        )

elif crop == "Maize":

    st.subheader("Maize yield estimation")

    st.write(
        "Our initial experiments found that the historical-mean "
        "baseline performed better than the tested machine-learning "
        "models on the maize test period."
    )

    split_index = int(len(df) * 0.8)
    historical_mean = df.iloc[:split_index]["maize_yield"].mean()

    st.metric(
        "Estimated maize yield",
        f"{historical_mean:,.2f}"
    )

    st.caption(
        "This is a historical-mean baseline, not an "
        "individual farm-level prediction."
    )

elif crop == "Potatoes":

    st.subheader("Potato yield estimation")

    st.write(
        "The previous-year yield baseline performed better "
        "than the tested machine-learning models."
    )

    previous_yield = st.number_input(
        "Enter previous-year potato yield",
        min_value=0.0,
        value=float(df["potato_yield_lag1"].median()),
        step=100.0
    )

    if st.button("Estimate potato yield"):

        st.success(
            f"Estimated potato yield: {previous_yield:,.2f}"
        )

        st.caption(
            "This estimate uses the previous year's yield "
            "as a simple baseline."
        )

st.divider()

st.subheader("About this project")

st.write(
    "AgriAI explores how machine learning, agricultural data "
    "and explainable AI can support climate-resilient "
    "agriculture and food-security decision-making in Kenya."
)

st.write(
    "**Research limitation:** The dataset contains 62 annual "
    "observations. The current results are preliminary and "
    "require validation with larger and geographically "
    "representative datasets."
)

st.markdown(
    "[View the AgriAI GitHub repository](https://github.com/"
    "mosesngethe686-eng/agriai-decision-support-assistant)"
)
