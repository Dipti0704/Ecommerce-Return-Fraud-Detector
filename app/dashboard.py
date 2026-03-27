import streamlit as st
import pandas as pd
import json
import os



st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- Load data ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "output", "user_fraud_scores.csv"))
    with open(os.path.join(BASE_DIR, "output", "dashboard_data.json")) as f:
        data = json.load(f)
    return df, data

df, data = load_data()

# --- Helper functions ---
def risk_badge(risk):
    if risk == "High":
        return "🔴 High"
    elif risk == "Medium":
        return "🟠 Medium"
    else:
        return "🟢 Low"

def action_recommendation(score):
    if score > 80:
        return "🚫 Block"
    elif score > 50:
        return "🧐 Review"
    else:
        return "✅ Allow"

# --- Title ---
st.title("🛒 AI Fraud Detection Dashboard")

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Users", data["summary"]["total_users"])
col2.metric("🛒 Orders", data["summary"]["total_orders"])
col3.metric("🔁 Returns", data["summary"]["total_returns"])
col4.metric("🚨 High Risk", data["summary"]["high_risk_count"])

st.divider()

# --- Risk Distribution ---
st.subheader("📊 Risk Distribution")
risk_df = pd.DataFrame.from_dict(data["risk_dist"], orient="index", columns=["count"])
st.bar_chart(risk_df)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

city_filter = st.sidebar.multiselect(
    "City",
    options=df["city"].unique(),
    default=df["city"].unique()
)

filtered_df = df[
    (df["risk_level"].isin(risk_filter)) &
    (df["city"].isin(city_filter))
].copy()

# Add badges + action
filtered_df["Risk Badge"] = filtered_df["risk_level"].apply(risk_badge)
filtered_df["Action"] = filtered_df["fraud_score"].apply(action_recommendation)

# --- Styled Table ---
st.subheader("🚨 User Risk Table")

st.dataframe(
    filtered_df.sort_values("fraud_score", ascending=False)[
        ["user_id", "city", "fraud_score", "Risk Badge", "Action", "flagged_reasons"]
    ],
    use_container_width=True
)

# --- Top Fraud Users ---
st.subheader("🔥 Top High-Risk Users")

top_users = df[df["risk_level"] == "High"] \
    .sort_values("fraud_score", ascending=False) \
    .head(10)

top_users["Action"] = top_users["fraud_score"].apply(action_recommendation)

st.dataframe(top_users, use_container_width=True)

# --- User Search ---
st.subheader("🔍 Search User")

user_id = st.text_input("Enter User ID")

if user_id:
    user_data = df[df["user_id"] == user_id]

    if not user_data.empty:
        row = user_data.iloc[0]

        st.success(f"User Found: {user_id}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Fraud Score", row["fraud_score"])
        col2.metric("Risk Level", row["risk_level"])
        col3.metric("Action", action_recommendation(row["fraud_score"]))

        st.write("### 🚨 Reasons")
        st.info(row["flagged_reasons"])

    else:
        st.error("User not found")

# --- Category Return Rates ---
st.subheader("📦 Category Return Rates")

cat_df = pd.DataFrame(
    list(data["category_return_rates"].items()),
    columns=["Category", "Return Rate"]
)

st.bar_chart(cat_df.set_index("Category"))

# --- Monthly Trend ---
st.subheader("📈 Monthly Order Trends")

trend_df = pd.DataFrame(data["monthly_orders"])
st.line_chart(trend_df)