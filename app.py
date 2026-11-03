
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
st.header("Explainable AI: Understanding the Predictions")

st.write(
    "Feature importance helps us understand which historical variables "
    "influenced the machine-learning models. These relationships do not "
    "establish cause and effect."
)

crop_files = {
    "Beans": "beans_permutation_importance.csv",
    "Maize": "maize_permutation_importance.csv",
    "Potatoes": "potatoes_permutation_importance.csv"
}

for crop_name, filename in crop_files.items():

    st.subheader(crop_name)

    try:
        importance_data = pd.read_csv(filename)

        st.write("Feature importance results")

        st.dataframe(
            importance_data,
            use_container_width=True
        )

        feature_col = next(
            (col for col in importance_data.columns
             if col.lower() == "feature"),
            importance_data.columns[0]
        )

        numeric_cols = importance_data.select_dtypes(
            include="number"
        ).columns.tolist()

        importance_col = next(
            (col for col in numeric_cols
             if "mean" in col.lower() or "importance" in col.lower()),
            numeric_cols[0]
        )

        chart_data = importance_data[
            [feature_col, importance_col]
        ].copy()

        chart_data = chart_data.dropna()

        chart_data = chart_data.sort_values(
            importance_col,
            ascending=True
        ).tail(10)

        st.write("Top 10 influential model features")

        st.bar_chart(
            chart_data.set_index(feature_col),
            horizontal=True
        )

    except FileNotFoundError:
        st.warning(
            f"The {crop_name.lower()} feature-importance file "
            "is not yet available."
        )

    except (IndexError, KeyError, ValueError) as error:
        st.error(
            f"Could not display the {crop_name} chart: {error}"
        )

st.caption(
    "Research limitation: Feature importance estimates are exploratory "
    "because the models were evaluated using only 13 test observations. "
    "Feature importance does not establish causation."
)
st.divider()

st.header("🤖 Ask AgriAI")
st.write(
    "Explore questions about crop-yield prediction, "
    "machine learning and climate-resilient agriculture. "
    "Answers are based on the current research findings."
)

questions = {
    "Why does previous-year yield matter for beans?":
        "The bean model identified previous-year bean yield "
        "as its most influential feature. This suggests that "
        "historical yield contains useful information for "
        "estimating subsequent yields. It does not prove causation.",

    "Why is temperature change important for maize?":
        "Temperature change was the leading feature in the "
        "maize permutation-importance analysis. This indicates "
        "that the tested model relied strongly on this variable. "
        "Further research is needed to establish its agricultural effects.",

    "Why does previous-year yield matter for potatoes?":
        "The potato model relied strongly on previous-year yield. "
        "The persistence baseline also performed better than the "
        "tested machine-learning models in the initial evaluation.",

    "What are the limitations of this research?":
        "The dataset contains 62 annual observations covering "
        "1962–2023. The test period contains only 13 observations. "
        "The results are exploratory and require validation using "
        "larger and geographically representative datasets.",

    "How can this research support food security?":
        "The research explores how historical agricultural data, "
        "machine learning and climate variables can contribute "
        "to crop-yield estimation. Further validation and integration "
        "with more geographically detailed data are needed before "
        "using the results for operational agricultural decisions."
}

selected_question = st.selectbox(
    "Choose a question",
    list(questions.keys())
)

if st.button("Get answer"):
    st.markdown("### AgriAI's response")
    st.write(questions[selected_question])

st.caption(
    "Research assistant prototype. Answers are predefined "
    "and grounded in the current study; this version does not "
    "yet use a live AI language model."
)
st.divider()

st.subheader("Ask your own question")

user_question = st.text_input(
    "What would you like to know about the research?",
    placeholder="e.g. What makes temperature important for maize?"
)

if st.button("Submit question"):

    question = user_question.lower()

    if not question.strip():
        st.warning("Please enter a question.")

    elif "bean" in question:
        st.write(
            "The bean model identified previous-year bean yield "
            "as its most influential feature. This suggests that "
            "historical yield contains useful information for "
            "estimating subsequent yields."
        )

    elif "maize" in question or "temperature" in question:
        st.write(
            "Temperature change was the leading feature in the "
            "maize permutation-importance analysis. This indicates "
            "that the tested model relied strongly on this variable. "
            "It does not establish a causal relationship."
        )

    elif "potato" in question:
        st.write(
            "Previous-year potato yield was highly influential. "
            "The previous-year baseline also outperformed the "
            "tested machine-learning models in the initial evaluation."
        )

    elif "limitation" in question or "data" in question:
        st.write(
            "The research uses 62 annual observations from 1962–2023. "
            "The test period contains only 13 observations. "
            "Further validation using larger datasets is necessary."
        )

    elif "food security" in question or "climate" in question:
        st.write(
            "AgriAI explores how agricultural data, machine learning "
            "and climate variables can support crop-yield estimation "
            "and food-security research in Kenya."
        )

    else:
        st.info(
            "This question is not yet covered by the research "
            "knowledge base. Try asking about beans, maize, potatoes, "
            "temperature, data limitations or food security."
        )

st.caption(
    "This is a prototype research assistant using predefined "
    "topic-based responses. It does not yet use generative AI."
)
