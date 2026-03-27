import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("output/user_fraud_scores.csv")
    with open("output/dashboard_data.json") as f:
        data = json.load(f)
    return df, data

df, data = load_data()

# --- Title ---
st.title("🛒 AI-Based Fraud Detection Dashboard")

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", data["summary"]["total_users"])
col2.metric("Total Orders", data["summary"]["total_orders"])
col3.metric("Total Returns", data["summary"]["total_returns"])
col4.metric("High Risk Users", data["summary"]["high_risk_count"])

st.divider()

# --- Risk Distribution ---
st.subheader("📊 Risk Distribution")

risk_df = pd.DataFrame.from_dict(data["risk_dist"], orient="index", columns=["count"])
st.bar_chart(risk_df)

# --- Filters ---
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

city_filter = st.sidebar.multiselect(
    "Select City",
    options=df["city"].unique(),
    default=df["city"].unique()
)

filtered_df = df[
    (df["risk_level"].isin(risk_filter)) &
    (df["city"].isin(city_filter))
]

# --- High Risk Users Table ---
st.subheader("🚨 User Risk Table")

st.dataframe(
    filtered_df.sort_values("fraud_score", ascending=False),
    use_container_width=True
)

# --- Top Fraud Users ---
st.subheader("🔥 Top High-Risk Users")

top_users = df[df["risk_level"] == "High"].sort_values("fraud_score", ascending=False).head(10)
st.dataframe(top_users, use_container_width=True)

# --- User Search ---
st.subheader("🔍 Search User")

user_id = st.text_input("Enter User ID")

if user_id:
    user_data = df[df["user_id"] == user_id]

    if not user_data.empty:
        st.success("User Found")

        st.write(user_data.T)

        st.subheader("🚨 Fraud Reasons")
        st.write(user_data["flagged_reasons"].values[0])
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
