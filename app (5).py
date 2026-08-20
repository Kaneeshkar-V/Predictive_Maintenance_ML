
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# ------------------------------------------------
# LOAD / GENERATE DATA
# ------------------------------------------------

np.random.seed(42)

n = 100

df = pd.DataFrame({
    "unit_id": [f"UNIT-{i+1:03d}" for i in range(n)],
    "vibration_mean": np.random.uniform(0.5, 5.0, n),
    "temperature_mean": np.random.uniform(40, 100, n),
    "gx": np.random.uniform(-2, 2, n),
    "gy": np.random.uniform(-2, 2, n),
    "gz": np.random.uniform(-2, 2, n)
})

# Simulated model output
df["risk_score"] = (
    df["vibration_mean"] * 15 +
    (df["temperature_mean"] - 40) * 0.5
).clip(0, 100)

df["health_score"] = (100 - df["risk_score"]).clip(0, 100)

df["risk_level"] = np.select(
    [
        df["risk_score"] >= 70,
        df["risk_score"] >= 40
    ],
    [
        "HIGH",
        "MODERATE"
    ],
    default="LOW"
)

df["prediction"] = np.select(
    [
        df["risk_score"] >= 70,
        df["risk_score"] >= 40
    ],
    [
        "FAILURE RISK",
        "WARNING"
    ],
    default="HEALTHY"
)

df["RUL_cycles"] = (
    200 * (1 - df["risk_score"] / 100)
).clip(0, 200)


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("⚙️ Predictive Maintenance AI")
st.caption(
    "Random Forest • Machine Health Monitoring • Fleet Risk Prediction"
)

st.divider()


# ------------------------------------------------
# KPI
# ------------------------------------------------

total = len(df)

healthy = int((df["risk_score"] < 40).sum())

at_risk = int((df["risk_score"] >= 40).sum())

avg_health = df["health_score"].mean()

avg_risk = df["risk_score"].mean()


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "TOTAL UNITS",
    total
)

c2.metric(
    "HEALTHY",
    healthy
)

c3.metric(
    "AT-RISK",
    at_risk
)

c4.metric(
    "AVERAGE HEALTH",
    f"{avg_health:.1f}/100"
)


st.divider()


# ------------------------------------------------
# FLEET OVERVIEW
# ------------------------------------------------

left, right = st.columns(2)


with left:

    st.subheader("📊 Fleet Health Overview")

    st.write(
        f"Average Health: **{avg_health:.1f}/100**"
    )

    st.progress(
        int(avg_health)
    )

    st.write(
        f"Average Risk: **{avg_risk:.1f}%**"
    )

    st.progress(
        int(avg_risk)
    )


with right:

    st.subheader("🚨 Risk Distribution")

    low = int(
        (df["risk_level"] == "LOW").sum()
    )

    moderate = int(
        (df["risk_level"] == "MODERATE").sum()
    )

    high = int(
        (df["risk_level"] == "HIGH").sum()
    )

    st.write(f"🟢 LOW: **{low}**")
    st.write(f"🟡 MODERATE: **{moderate}**")
    st.write(f"🔴 HIGH: **{high}**")


st.divider()


# ------------------------------------------------
# TOP RISK UNITS
# ------------------------------------------------

st.subheader("⚠️ Alert Priority Queue")

top = df.sort_values(
    "risk_score",
    ascending=False
).head(10)

st.dataframe(
    top[
        [
            "unit_id",
            "risk_score",
            "health_score",
            "risk_level",
            "RUL_cycles"
        ]
    ].rename(
        columns={
            "unit_id": "Unit",
            "risk_score": "Risk %",
            "health_score": "Health",
            "risk_level": "Risk Level",
            "RUL_cycles": "RUL Cycles"
        }
    ),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ------------------------------------------------
# LIVE PREDICTION
# ------------------------------------------------

st.subheader("🔮 Live Machine Prediction")

unit = st.selectbox(
    "Select Machine",
    df["unit_id"]
)

vibration = st.slider(
    "Vibration",
    0.0,
    5.0,
    1.0,
    0.1
)

if st.button(
    "🚀 RUN PREDICTION",
    type="primary"
):

    risk = min(
        100,
        vibration * 20
    )

    health = 100 - risk

    if risk >= 70:
        level = "HIGH"
        decision = "🔴 SCHEDULE MAINTENANCE"

    elif risk >= 40:
        level = "MODERATE"
        decision = "🟡 INSPECT MACHINE"

    else:
        level = "LOW"
        decision = "🟢 MACHINE HEALTHY"

    rul = max(
        0,
        int(200 * (1 - risk / 100))
    )

    st.success("Prediction completed!")

    a, b, c, d = st.columns(4)

    a.metric(
        "RISK",
        f"{risk:.1f}%"
    )

    b.metric(
        "HEALTH",
        f"{health:.1f}/100"
    )

    c.metric(
        "RISK LEVEL",
        level
    )

    d.metric(
        "EST. RUL",
        f"{rul} cycles"
    )

    st.warning(
        f"Maintenance Decision: {decision}"
    )


st.divider()


# ------------------------------------------------
# RECENT PREDICTIONS
# ------------------------------------------------

st.subheader("🧠 Recent Predictions")

st.dataframe(
    df.head(15)[
        [
            "unit_id",
            "health_score",
            "risk_score",
            "prediction",
            "risk_level",
            "RUL_cycles"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Predictive Maintenance AI • Random Forest • "
    "Synthetic Sensor Dataset"
)
