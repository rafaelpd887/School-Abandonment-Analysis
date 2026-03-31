import pandas as pd

# ---------------- CONFIG ---------------- #

PREDICTORS = [
    "localizacao",
    "rede",
    "atu_em",
    "had_em",
    "tdi_em",
    "taxa_aprovacao_em",
    "taxa_reprovacao_em",
    "dsu_em",
    "afd_em_grupo_1",
]

TARGET = "taxa_abandono_em"


# ---------------- TRAINING ---------------- #
def clean_data_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning for model training.

    - Keeps target column
    - Drops rows with missing target values
    """

    df = df.copy()

    # Select relevant columns
    df = df[PREDICTORS + [TARGET]]

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Convert categorical features
    df["localizacao"] = df["localizacao"].astype("category")
    df["rede"] = df["rede"].astype("category")

    return df


# ---------------- INFERENCE ---------------- #
def clean_data_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning for inference (API / dashboard).

    - Uses only feature columns
    - Does NOT drop rows
    """

    df = df.copy()

    # Keep only predictors
    df = df[PREDICTORS]

    # Convert categorical features
    df["localizacao"] = df["localizacao"].astype("category")
    df["rede"] = df["rede"].astype("category")

    return df


# ---------------- LOAD RAW DATA ---------------- #
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset (supports .csv and .csv.gz)
    """

    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


# ---------------- SCRIPT ENTRYPOINT ---------------- #
if __name__ == "__main__":
    """
    Run full data cleaning pipeline and save processed dataset
    """

    df = load_raw_data(
        "data/raw/br_inep_indicadores_educacionais_brasil.csv.gz"
    )

    df_clean = clean_data_train(df)

    df_clean.to_csv("data/processed/data.csv", index=False)

    print("Processed dataset saved to data/processed/data.csv")

# Variables selected to capture school characteristics, student performance history,
# and teacher quality, while avoiding target leakage.

## Analysis of predictors:
# localizacao → makes sense. Schools in different regions may have different dropout rates. ✅
# rede → makes sense. Public, private, or other school networks directly influence resources and dropout. ✅
# atu_em → average students per class. Makes sense: larger classes may increase dropout risk. ✅
# had_em → daily teaching hours. Makes sense: very high or very low workload can impact dropout. ✅
# tdi_em → age-grade distortion rate. Makes sense: if many students are behind, risk of dropout increases. ✅
# taxa_aprovacao_em / taxa_reprovacao_em → makes sense to include, but be careful: may be highly correlated with dropout rate. Depending on the model, can cause target leakage. ⚠️
# tnr_em → non-response rate (from survey). Lower relevance, but may indicate data quality. Optional. ⚠️
# dsu_em → percentage of teachers with higher education. Makes sense: more qualified teachers can reduce dropout. ✅
# afd_em_grupo_1-5 → teacher training adequacy. Makes sense, but you may not need all groups separately:
#     consider using only group 1 (fully adequate teachers) or an aggregated metric (group 1 + group 2)
#     to simplify and avoid multicollinearity. ⚠️