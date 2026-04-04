# app_full.py
import sys
import os
import streamlit as st
import pandas as pd
import joblib

# ADD PROJECT ROOT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_cleaning import clean_data_inference
from scripts.feature_engineering import preprocess

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="School Dropout Prediction Dashboard",
    layout="wide"
)

st.title("📊 School Dropout Prediction Dashboard")
st.markdown("Predicting and analyzing school dropout rates using Machine Learning")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

# ---------------- LOAD ORIGINAL DATA ---------------- #
@st.cache_data
def load_raw_data():
    return pd.read_csv(
        "data/raw/br_inep_indicadores_educacionais_brasil.csv.gz",
        compression='gzip'
    )

df_raw = load_raw_data()

# ---------------- PROCESSING + PREDICTION ---------------- #
X_all = preprocess(clean_data_inference(df_raw))
df_raw["predicted_dropout"] = model.predict(X_all)

# ---------------- FINAL COLUMN ---------------- #
df_raw["final_dropout"] = df_raw["taxa_abandono_em"]
df_raw.loc[df_raw["final_dropout"].isna(), "final_dropout"] = df_raw["predicted_dropout"]

# ---------------- SIDEBAR - FILTERS ---------------- #
st.sidebar.header("🔎 Filters")
filtered_df = df_raw.copy()

# Rede
if "rede" in df_raw.columns:
    redes = df_raw["rede"].dropna().unique()
    selected_rede = st.sidebar.multiselect(
        "School Network",
        options=redes,
        default=redes
    )
    filtered_df = filtered_df[
        (filtered_df["rede"].isin(selected_rede)) | (filtered_df["rede"].isna())
    ]

# Localização
if "localizacao" in df_raw.columns:
    locs = df_raw["localizacao"].dropna().unique()
    selected_loc = st.sidebar.multiselect(
        "Location",
        options=locs,
        default=locs
    )
    filtered_df = filtered_df[
        (filtered_df["localizacao"].isin(selected_loc)) | (filtered_df["localizacao"].isna())
    ]

# ---------------- METRICS ---------------- #
st.subheader("📈 Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Avg Dropout", round(filtered_df["final_dropout"].mean(), 4))
col2.metric("Max Dropout", round(filtered_df["final_dropout"].max(), 4))
col3.metric("Min Dropout", round(filtered_df["final_dropout"].min(), 4))

# ---------------- CHARTS ---------------- #
st.subheader("📊 Dropout Distribution")

dropout_counts = filtered_df["final_dropout"].round(2).value_counts().sort_index()
st.bar_chart(dropout_counts)

# ---------------- TOP SCHOOLS ---------------- #
st.subheader("🏫 Top 10 Schools with Highest Dropout")

top10 = filtered_df.sort_values("final_dropout", ascending=False).head(10)

cols = [
    "localizacao",
    "rede",
    "atu_em",
    "had_em",
    "tdi_em",
    "taxa_aprovacao_em",
    "taxa_reprovacao_em",
    "dsu_em",
    "afd_em_grupo_1",
    "taxa_abandono_em",
    "predicted_dropout"
]

cols = [col for col in cols if col in top10.columns]

top10_display = top10[cols].rename(columns={
    "localizacao": "Location",
    "rede": "Network",
    "atu_em": "Attendance",
    "had_em": "Dropout History",
    "tdi_em": "Age-Grade Distortion",
    "taxa_aprovacao_em": "Approval Rate",
    "taxa_reprovacao_em": "Failure Rate",
    "dsu_em": "Socioeconomic Index",
    "afd_em_grupo_1": "Teacher Qualification",
    "taxa_abandono_em": "Actual Dropout",
    "predicted_dropout": "Predicted Dropout"
})

st.dataframe(top10_display)

# ---------------- DATA PREVIEW ---------------- #
st.subheader("📋 Dataset Preview")
st.dataframe(filtered_df)

# ---------------- DOWNLOAD ---------------- #
st.subheader("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="full_predictions.csv",
    mime="text/csv",
)