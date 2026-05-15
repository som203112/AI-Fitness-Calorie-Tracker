import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# =========================
# LOAD MODEL
# =========================

model = joblib.load("app/model.pkl")

# =========================
# LOAD DATA
# =========================

calories_data = pd.read_csv("data/calories.csv")
exercise_data = pd.read_csv("data/exercise.csv")

data = exercise_data.merge(calories_data, on="User_ID")
X = data.drop(columns=["User_ID", "Calories"])

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Fitness Calorie Tracker",
    page_icon="🔥",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #1e3a2f;
        color: white;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================

st.title("🔥 AI Fitness Calorie Tracker")

st.markdown(
    """
    Predict calories burned using Machine Learning and exercise data.
    """
)

st.write("---")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("About")

st.sidebar.info(
    """
    This AI-powered application predicts calories burned during exercise
    using physiological and workout parameters.
    """
)

st.sidebar.write("Built using:")
st.sidebar.write("- Python")
st.sidebar.write("- XGBoost")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Scikit-learn")

# =========================
# INPUTS
# =========================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    10,
    80,
    25
)

height = st.slider(
    "Height (cm)",
    100,
    250,
    170
)

weight = st.slider(
    "Weight (kg)",
    30,
    200,
    70
)

duration = st.slider(
    "Exercise Duration (minutes)",
    1,
    120,
    30
)

heart_rate = st.slider(
    "Heart Rate",
    60,
    200,
    100
)

body_temp = st.slider(
    "Body Temperature (°C)",
    36.0,
    42.0,
    37.0
)

# =========================
# BMI CALCULATION
# =========================

height_m = height / 100
bmi = weight / (height_m ** 2)

st.metric(
    label="BMI",
    value=f"{bmi:.2f}"
)

# =========================
# PREPROCESS INPUT
# =========================

gender_value = 0 if gender == "Male" else 1

input_data = np.array([
    [
        gender_value,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]
])

# =========================
# PREDICTION
# =========================

if st.button("Predict Calories Burned"):

    prediction = model.predict(input_data)

    st.markdown(
        f'''
        <div class="prediction-box">
            🔥 Estimated Calories Burned<br><br>
            {prediction[0]:.2f} kcal
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Health Insights
    st.write("")

    if prediction[0] < 200:
        st.info("Light calorie burn activity.")
    elif prediction[0] < 400:
        st.success("Moderate workout intensity.")
    else:
        st.warning("High calorie burn workout!")

    st.balloons()

# =========================
# DATA VISUALIZATION
# =========================

st.write("---")

st.header("📊 Fitness Data Insights")

# Calories Distribution
fig1 = px.histogram(
    data,
    x="Calories",
    nbins=40,
    title="Calories Burned Distribution"
)

st.plotly_chart(fig1)

# Duration vs Calories
fig2 = px.scatter(
    data,
    x="Duration",
    y="Calories",
    color="Gender",
    title="Exercise Duration vs Calories Burned"
)

st.plotly_chart(fig2)

# Heart Rate vs Calories
fig3 = px.scatter(
    data,
    x="Heart_Rate",
    y="Calories",
    color="Gender",
    title="Heart Rate vs Calories Burned"
)

st.plotly_chart(fig3)


# =========================
# FEATURE IMPORTANCE
# =========================

st.write("---")

st.header("📌 Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig4 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance Analysis"
)

st.plotly_chart(fig4)

# =========================
# FOOTER
# =========================

st.write("---")

st.caption(
    "Built by Soham Kadam"
)
