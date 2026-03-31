import sys
import os

# ADD PROJECT ROOT TO PATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib

from scripts.data_cleaning import clean_data_inference
from scripts.feature_engineering import preprocess

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="School Dropout Prediction",
    layout="wide"
)

st.title("📊 School Dropout Prediction Dashboard")
st.markdown("Predicting and analyzing school dropout rates using Machine Learning")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data.csv")

df = load_data()

# ---------------- PROCESS DATA ---------------- #

df_model = clean_data_inference(df)
X = preprocess(df_model)

df["predicted_dropout"] = model.predict(X)

if "taxa_abandono_em" in df.columns:
    df["final_dropout"] = df["taxa_abandono_em"].fillna(df["predicted_dropout"])
else:
    df["final_dropout"] = df["predicted_dropout"]

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("🔎 Filters")

filtered_df = df.copy()

if "rede" in df.columns:
    redes = df["rede"].dropna().unique()
    selected_rede = st.sidebar.multiselect(
        "School Network",
        options=redes,
        default=redes
    )
    filtered_df = filtered_df[filtered_df["rede"].isin(selected_rede)]

if "localizacao" in df.columns:
    locs = df["localizacao"].dropna().unique()
    selected_loc = st.sidebar.multiselect(
        "Location",
        options=locs,
        default=locs
    )
    filtered_df = filtered_df[filtered_df["localizacao"].isin(selected_loc)]

# ---------------- METRICS ---------------- #

st.subheader("📈 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Dropout", round(filtered_df["final_dropout"].mean(), 4))
col2.metric("Max Dropout", round(filtered_df["final_dropout"].max(), 4))
col3.metric("Min Dropout", round(filtered_df["final_dropout"].min(), 4))

# ---------------- CHARTS ---------------- #

st.subheader("📊 Dropout Distribution")

st.bar_chart(filtered_df["final_dropout"].value_counts().sort_index())

# ---------------- TOP SCHOOLS ---------------- #

st.subheader("🏫 Top 10 Schools with Highest Dropout")

top10 = filtered_df.sort_values("final_dropout", ascending=False).head(10)
st.dataframe(top10)

# ---------------- DATA PREVIEW ---------------- #

st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df.head(50))

# ---------------- DOWNLOAD ---------------- #

st.subheader("⬇️ Download Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_predictions.csv",
    mime="text/csv",
)