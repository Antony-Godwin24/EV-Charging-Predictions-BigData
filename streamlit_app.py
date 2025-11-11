import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="⚡ EV Charging Predictions Dashboard", layout="wide", page_icon="🔋")

# -----------------------------
# Theme & Appearance
# -----------------------------
st.sidebar.header("🎨 Theme & View Options")
appearance = st.sidebar.radio("Appearance", ["Light", "Dark"], index=0)

GRADIENTS_LIGHT = {
    "Aurora (teal→mint→violet)": "linear-gradient(180deg, #10b981 0%, #99f6e4 45%, #a78bfa 100%)",
    "Sunset (coral→gold→plum)": "linear-gradient(180deg, #fb7185 0%, #fbbf24 45%, #a78bfa 100%)",
    "Lagoon (aqua→sky→indigo)": "linear-gradient(180deg, #34d399 0%, #60a5fa 50%, #6366f1 100%)",
}
GRADIENTS_DARK = {
    "Neon (teal→purple)": "linear-gradient(180deg, #0f766e 0%, #7c3aed 100%)",
    "Cosmos (navy→violet)": "linear-gradient(180deg, #0f172a 0%, #312e81 100%)",
    "Ember (charcoal→amber)": "linear-gradient(180deg, #111827 0%, #b45309 100%)",
}

grad_choice = st.sidebar.selectbox("Gradient", list(GRADIENTS_LIGHT.keys()) if appearance == "Light" else list(GRADIENTS_DARK.keys()))
GRADIENT = (GRADIENTS_LIGHT if appearance == "Light" else GRADIENTS_DARK)[grad_choice]

TEXT_COLOR = "#0f172a" if appearance == "Light" else "#f1f5f9"
CARD_BG = "rgba(255,255,255,0.92)" if appearance == "Light" else "rgba(17,24,39,0.85)"
BORDER = "#e2e8f0" if appearance == "Light" else "#334155"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: {GRADIENT};
}}
.block-container {{
  color:{TEXT_COLOR};
  background: {CARD_BG};
  border-radius: 18px;
  padding: 1.2rem 1.4rem 2rem;
  border: 1px solid {BORDER};
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}}
h1, h2, h3, label, .stMarkdown, .stDataFrame {{ color:{TEXT_COLOR}; }}
.metric-card {{
  background:rgba(248,250,252,0.6);
  border:1px solid {BORDER};
  padding:.75rem 1rem;
  border-radius:10px;
  text-align:center;
}}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

# -----------------------------
# Load Model + Dataset
# -----------------------------
DATA_PATH = "dataset/ev_hourly_agg.csv"
MODEL_PATH = "model/ev_load_model.joblib"

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.error("❌ Required files missing. Run `python spark_app.py` first to train the model.")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

st.title("🔋 EV Charging Load Prediction Dashboard")
st.caption("Built with PySpark + Streamlit | Inspired by Krish-CS style ✨")

# -----------------------------
# Summary Cards
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'>📅 Records<br><b>{len(df):,}</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'>⚡ Avg Load (kW)<br><b>{df['avg_load'].mean():.2f}</b></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'>📈 Max Load (kW)<br><b>{df['avg_load'].max():.2f}</b></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Chart 1 – Hourly Load Trend
# -----------------------------
st.subheader("⏱️ Hourly Average Charging Load Trend")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x=range(len(df)), y=df["avg_load"], color="#0ea5e9", linewidth=1.5)
ax.set_xlabel("Time Index (Hour)")
ax.set_ylabel("Average Load (kW)")
ax.set_title("Hourly Average Load")
st.pyplot(fig)

# -----------------------------
# Chart 2 – Load Distribution by Hour of Day
# -----------------------------
if "hour" in df.columns:
    st.subheader("🕐 Load Distribution by Hour of Day")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["hour"], y=df["avg_load"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Load (kW)")
    st.pyplot(fig)

# -----------------------------
# Chart 3 – Load by Weekday
# -----------------------------
if "weekday" in df.columns:
    st.subheader("📆 Load Variation Across Weekdays")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=df["weekday"], y=df["avg_load"], palette="crest", ax=ax)
    ax.set_xlabel("Weekday (1=Sunday)")
    ax.set_ylabel("Average Load (kW)")
    st.pyplot(fig)

# -----------------------------
# Chart 4 – Feature Importance
# -----------------------------
st.subheader("🌲 Feature Importance (Random Forest)")
features = ["load_lag1", "load_lag24", "hour", "weekday"]
try:
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
except Exception:
    st.warning("Feature importances not available for this model.")

# -----------------------------
# Chart 5 – Next 24h Prediction
# -----------------------------
st.subheader("🔮 Predicted Charging Load for Next 24 Hours")

# Predict using last 24 hours’ lag values
latest = df.tail(24).copy()
X_pred = latest[["load_lag1", "load_lag24", "hour", "weekday"]]
preds = model.predict(X_pred)

pred_df = pd.DataFrame({
    "Hour Ahead": np.arange(1, 25),
    "Predicted_Load_kW": preds
})

fig, ax = plt.subplots(figsize=(9, 4))
sns.lineplot(data=pred_df, x="Hour Ahead", y="Predicted_Load_kW", marker="o", linewidth=2, color="#6366f1")
ax.set_title("Forecasted EV Charging Load (Next 24 Hours)")
ax.set_ylabel("Predicted Load (kW)")
st.pyplot(fig)
st.dataframe(pred_df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("💡 Project by Antony Godwin with 💖 | Theme inspired by Krish-CS | Powered by PySpark, scikit-learn & Streamlit")
