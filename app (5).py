import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "predictive_maintenance_models.joblib"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "metrics.json"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "training_dataset.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found.\n\n"
            "Run `python train_model.py` first."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


bundle = load_models()

features = bundle["features"]

classifier = bundle["classifier"]
rul_model = bundle["rul_model"]
anomaly_model = bundle["anomaly_model"]


# ============================================================
# LOAD METRICS
# ============================================================

if os.path.exists(METRICS_PATH):

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

else:
    metrics = {}


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "training_dataset.csv not found.\n\n"
        "Run `python train_model.py` first."
    )

    st.stop()


df = pd.read_csv(DATA_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_score(model, X):

    """
    Converts Random Forest class probabilities
    into a model-derived risk score.

    HEALTHY      = 0 risk
    WARNING      = 50 risk
    FAILURE RISK = 100 risk
    """

    probabilities = model.predict_proba(X)

    classes = list(model.named_steps["model"].classes_)

    probability_df = pd.DataFrame(
        probabilities,
        columns=classes
    )

    healthy_probability = (
        probability_df["HEALTHY"]
        if "HEALTHY" in probability_df
        else 0
    )

    warning_probability = (
        probability_df["WARNING"]
        if "WARNING" in probability_df
        else 0
    )

    failure_probability = (
        probability_df["FAILURE RISK"]
        if "FAILURE RISK" in probability_df
        else 0
    )

    risk = (
        warning_probability * 50
        + failure_probability * 100
    )

    risk = np.clip(
        risk,
        0,
        100
    )

    confidence = probability_df.max(
        axis=1
    ) * 100

    return (
        risk,
        confidence,
        probability_df
    )


def risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MODERATE"

    return "LOW"


def maintenance_decision(level):

    if level == "HIGH":
        return "🔴 SCHEDULE MAINTENANCE"

    elif level == "MODERATE":
        return "🟡 INSPECT MACHINE"

    return "🟢 MACHINE HEALTHY"


# ============================================================
# FLEET PREDICTION
# ============================================================

X_fleet = df[features]

fleet_prediction = classifier.predict(
    X_fleet
)

fleet_rul = rul_model.predict(
    X_fleet
)

fleet_rul = np.clip(
    fleet_rul,
    0,
    None
)

fleet_risk, fleet_confidence, fleet_probabilities = (
    get_risk_score(
        classifier,
        X_fleet
    )
)

fleet_anomaly = anomaly_model.predict(
    X_fleet
)

df["prediction"] = fleet_prediction

df["risk_score"] = fleet_risk

df["confidence"] = fleet_confidence

df["health_score"] = (
    100 - df["risk_score"]
).clip(
    0,
    100
)

df["risk_level"] = [
    risk_level(x)
    for x in df["risk_score"]
]

df["RUL_cycles"] = fleet_rul

df["anomaly"] = np.where(
    fleet_anomaly == -1,
    "ANOMALY",
    "NORMAL"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "⚙️ Predictive Maintenance AI"
)

st.caption(
    "Random Forest Classification • "
    "Random Forest RUL Regression • "
    "Isolation Forest Anomaly Detection"
)

st.divider()


# ============================================================
# KPI SECTION
# ============================================================

total_units = len(df)

healthy_units = int(
    (df["risk_score"] < 40).sum()
)

at_risk_units = int(
    (df["risk_score"] >= 40).sum()
)

average_health = float(
    df["health_score"].mean()
)

average_risk = float(
    df["risk_score"].mean()
)

anomaly_count = int(
    (df["anomaly"] == "ANOMALY").sum()
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "TOTAL UNITS",
    total_units
)

c2.metric(
    "HEALTHY",
    healthy_units
)

c3.metric(
    "AT-RISK",
    at_risk_units
)

c4.metric(
    "AVERAGE HEALTH",
    f"{average_health:.1f}/100"
)

c5.metric(
    "ANOMALIES",
    anomaly_count
)


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🧠 Model Performance")

m1, m2, m3, m4 = st.columns(4)


if metrics:

    m1.metric(
        "CLASSIFICATION ACCURACY",
        f"{metrics.get('classification_accuracy', 0) * 100:.1f}%"
    )

    m2.metric(
        "F1 SCORE",
        f"{metrics.get('classification_f1', 0) * 100:.1f}%"
    )

    m3.metric(
        "RUL MAE",
        f"{metrics.get('rul_mae_cycles', 0):.1f} cycles"
    )

    m4.metric(
        "RUL R²",
        f"{metrics.get('rul_r2', 0):.3f}"
    )

else:

    m1.metric(
        "CLASSIFICATION ACCURACY",
        "N/A"
    )

    m2.metric(
        "F1 SCORE",
        "N/A"
    )

    m3.metric(
        "RUL MAE",
        "N/A"
    )

    m4.metric(
        "RUL R²",
        "N/A"
    )


st.caption(
    "Metrics shown above are evaluation results from the "
    "training/test split used during model training."
)


st.divider()


# ============================================================
# FLEET HEALTH OVERVIEW
# ============================================================

left, right = st.columns(2)


with left:

    st.subheader(
        "📊 Fleet Health Overview"
    )

    st.write(
        f"Average Health: "
        f"**{average_health:.1f}/100**"
    )

    st.progress(
        int(np.clip(
            average_health,
            0,
            100
        ))
    )

    st.write(
        f"Average Risk: "
        f"**{average_risk:.1f}%**"
    )

    st.progress(
        int(np.clip(
            average_risk,
            0,
            100
        ))
    )


with right:

    st.subheader(
        "🚨 Risk Distribution"
    )

    low_count = int(
        (df["risk_level"] == "LOW").sum()
    )

    moderate_count = int(
        (df["risk_level"] == "MODERATE").sum()
    )

    high_count = int(
        (df["risk_level"] == "HIGH").sum()
    )

    st.write(
        f"🟢 LOW: **{low_count}**"
    )

    st.write(
        f"🟡 MODERATE: **{moderate_count}**"
    )

    st.write(
        f"🔴 HIGH: **{high_count}**"
    )


st.divider()


# ============================================================
# RISK DISTRIBUTION CHART
# ============================================================

st.subheader(
    "📈 Risk Distribution"
)

risk_chart = pd.DataFrame(
    {
        "Risk Level": [
            "LOW",
            "MODERATE",
            "HIGH"
        ],
        "Machines": [
            low_count,
            moderate_count,
            high_count
        ]
    }
)

st.bar_chart(
    risk_chart.set_index(
        "Risk Level"
    )
)


st.divider()


# ============================================================
# ALERT PRIORITY QUEUE
# ============================================================

st.subheader(
    "⚠️ Alert Priority Queue"
)

top = (
    df.sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
    .copy()
)

top_display = top[
    [
        "unit_id",
        "risk_score",
        "health_score",
        "prediction",
        "risk_level",
        "RUL_cycles",
        "anomaly"
    ]
].copy()


top_display.columns = [
    "Unit",
    "Risk %",
    "Health",
    "Prediction",
    "Risk Level",
    "RUL Cycles",
    "Anomaly"
]


top_display["Risk %"] = (
    top_display["Risk %"]
    .round(1)
)

top_display["Health"] = (
    top_display["Health"]
    .round(1)
)

top_display["RUL Cycles"] = (
    top_display["RUL Cycles"]
    .round(1)
)


st.dataframe(
    top_display,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# LIVE MACHINE PREDICTION
# ============================================================

st.subheader(
    "🔮 Live Machine Prediction"
)


machine = st.selectbox(
    "Select Machine",
    df["unit_id"].tolist()
)


st.markdown(
    "### Sensor Inputs"
)


input_col1, input_col2, input_col3 = st.columns(3)

input_col4, input_col5, input_col6 = st.columns(3)

input_col7, input_col8, input_col9 = st.columns(3)


with input_col1:

    vibration_mean = st.number_input(
        "Vibration Mean",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


with input_col2:

    vibration_max = st.number_input(
        "Vibration Max",
        min_value=0.0,
        max_value=15.0,
        value=1.3,
        step=0.1
    )


with input_col3:

    vibration_rms = st.number_input(
        "Vibration RMS",
        min_value=0.0,
        max_value=15.0,
        value=1.1,
        step=0.1
    )


with input_col4:

    vibration_std = st.number_input(
        "Vibration Std",
        min_value=0.0,
        max_value=10.0,
        value=0.15,
        step=0.01
    )


with input_col5:

    temperature = st.number_input(
        "Temperature °C",
        min_value=0.0,
        max_value=150.0,
        value=35.0,
        step=0.5
    )


with input_col6:

    current = st.number_input(
        "Current A",
        min_value=0.0,
        max_value=10.0,
        value=0.8,
        step=0.05
    )


with input_col7:

    rpm = st.number_input(
        "RPM",
        min_value=0.0,
        max_value=5000.0,
        value=3000.0,
        step=50.0
    )


with input_col8:

    gyro_rms = st.number_input(
        "Gyro RMS",
        min_value=0.0,
        max_value=10.0,
        value=0.15,
        step=0.01
    )


with input_col9:

    st.info(
        "Live hardware data can replace these values "
        "later using ESP32 + sensors."
    )


st.write("")


if st.button(
    "🚀 RUN ML PREDICTION",
    type="primary",
    use_container_width=True
):

    live_data = pd.DataFrame(
        [
            {
                "vibration_mean":
                    vibration_mean,

                "vibration_max":
                    vibration_max,

                "vibration_rms":
                    vibration_rms,

                "vibration_std":
                    vibration_std,

                "temperature_mean":
                    temperature,

                "current":
                    current,

                "rpm":
                    rpm,

                "gyro_rms":
                    gyro_rms
            }
        ]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    predicted_status = classifier.predict(
        live_data
    )[0]


    risk_array, confidence_array, probabilities = (
        get_risk_score(
            classifier,
            live_data
        )
    )


    risk = float(
        risk_array[0]
    )

    confidence = float(
        confidence_array[0]
    )


    health = float(
        np.clip(
            100 - risk,
            0,
            100
        )
    )


    level = risk_level(
        risk
    )


    decision = maintenance_decision(
        level
    )


    # ========================================================
    # RUL
    # ========================================================

    predicted_rul = float(
        rul_model.predict(
            live_data
        )[0]
    )

    predicted_rul = max(
        0,
        predicted_rul
    )


    # ========================================================
    # ANOMALY
    # ========================================================

    anomaly_prediction = anomaly_model.predict(
        live_data
    )[0]

    anomaly_score = float(
        anomaly_model.decision_function(
            live_data
        )[0]
    )


    if anomaly_prediction == -1:

        anomaly_text = "⚠️ ANOMALY DETECTED"

    else:

        anomaly_text = "✅ NORMAL PATTERN"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success(
        f"Prediction completed for {machine}"
    )


    r1, r2, r3, r4, r5 = st.columns(5)


    r1.metric(
        "RISK",
        f"{risk:.1f}%"
    )

    r2.metric(
        "HEALTH",
        f"{health:.1f}/100"
    )

    r3.metric(
        "RISK LEVEL",
        level
    )

    r4.metric(
        "EST. RUL",
        f"{predicted_rul:.1f} cycles"
    )

    r5.metric(
        "MODEL CONFIDENCE",
        f"{confidence:.1f}%"
    )


    st.info(
        f"### Prediction: {predicted_status}"
    )


    if level == "HIGH":

        st.error(
            f"Maintenance Decision: {decision}"
        )

    elif level == "MODERATE":

        st.warning(
            f"Maintenance Decision: {decision}"
        )

    else:

        st.success(
            f"Maintenance Decision: {decision}"
        )


    st.write(
        f"### Anomaly Detection: {anomaly_text}"
    )


    st.write(
        f"Isolation Forest decision score: "
        f"**{anomaly_score:.4f}**"
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Model Class Probabilities"
    )


    probability_display = probabilities.iloc[0].copy()

    probability_display = (
        probability_display
        .rename(
            {
                "HEALTHY":
                    "Healthy",

                "WARNING":
                    "Warning",

                "FAILURE RISK":
                    "Failure Risk"
            }
        )
        * 100
    )


    st.bar_chart(
        probability_display
    )


    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    st.subheader(
        "📋 Input Sensor Summary"
    )

    summary = pd.DataFrame(
        {
            "Feature": features,

            "Value": [
                vibration_mean,
                vibration_max,
                vibration_rms,
                vibration_std,
                temperature,
                current,
                rpm,
                gyro_rms
            ]
        }
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# RECENT / FLEET PREDICTIONS
# ============================================================

st.subheader(
    "🧠 Machine Prediction Table"
)


recent = df[
    [
        "unit_id",
        "health_score",
        "risk_score",
        "prediction",
        "risk_level",
        "confidence",
        "RUL_cycles",
        "anomaly"
    ]
].head(20).copy()


recent.columns = [
    "Unit",
    "Health",
    "Risk %",
    "Prediction",
    "Risk Level",
    "Confidence",
    "RUL Cycles",
    "Anomaly"
]


recent["Health"] = (
    recent["Health"]
    .round(1)
)

recent["Risk %"] = (
    recent["Risk %"]
    .round(1)
)

recent["Confidence"] = (
    recent["Confidence"]
    .round(1)
)

recent["RUL Cycles"] = (
    recent["RUL Cycles"]
    .round(1)
)


st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# RUL DISTRIBUTION
# ============================================================

st.subheader(
    "⏳ Remaining Useful Life Distribution"
)

rul_chart = (
    df["RUL_cycles"]
    .clip(lower=0)
)

st.bar_chart(
    rul_chart
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Predictive Maintenance AI | "
    "Random Forest Classifier + "
    "Random Forest RUL Regressor + "
    "Isolation Forest | "
    "Synthetic Sensor Dataset"
)
