import streamlit as st
import pandas as pd
import joblib

from scripts.data_cleaning import clean_data_inference
from scripts.feature_engineering import preprocess

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="School Dropout Prediction", layout="wide")

st.title("📊 School Dropout Prediction Dashboard")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    return pd.read_pickle("data/processed/data.pkl")

df = load_data()

# ---------------- PROCESS DATA ---------------- #

df_model = clean_data_inference(df)
X = preprocess(df_model)

# ---------------- PREDICTIONS ---------------- #

df["predicted_dropout"] = model.predict(X)

# opcional (se quiser combinar real + previsto)
if "taxa_abandono_em" in df.columns:
    df["final_dropout"] = df["taxa_abandono_em"].fillna(df["predicted_dropout"])

# ---------------- UI ---------------- #

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dropout Distribution")
st.hist_chart(df["predicted_dropout"])

st.subheader("Top 10 Schools with Highest Predicted Dropout")

top10 = df.sort_values("predicted_dropout", ascending=False).head(10)
st.dataframe(top10)

# ---------------- FILTER ---------------- #

st.sidebar.header("Filters")

if "rede" in df.columns:
    selected_rede = st.sidebar.selectbox(
        "Select School Network",
        options=df["rede"].dropna().unique()
    )

    filtered_df = df[df["rede"] == selected_rede]

    st.subheader(f"Filtered Data - {selected_rede}")
    st.dataframe(filtered_df.head())

# ---------------- DOWNLOAD ---------------- #

st.subheader("Download Predictions")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
)

