import pandas as pd

# =========================
# Feature Engineering
# =========================
def preprocess(df):
    df = df.copy()  
    
    categorical_cols = ['localizacao', 'rede']

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Ratios / interactions
    df['reprovacao_per_had'] = df['taxa_reprovacao_em'] / (df['had_em'] + 1)
    df['dsu_per_afd'] = df['dsu_em'] / (df['afd_em_grupo_1'] + 1)

    # Differences and sums
    df['had_minus_afd'] = df['had_em'] - df['afd_em_grupo_1']
    df['had_plus_afd'] = df['had_em'] + df['afd_em_grupo_1']

    # Products for interactions
    df['reprovacao_x_dsu'] = df['taxa_reprovacao_em'] * df['dsu_em']
    df['reprovacao_x_afd'] = df['taxa_reprovacao_em'] * df['afd_em_grupo_1']
    df['dsu_x_afd'] = df['dsu_em'] * df['afd_em_grupo_1']

    return df


# =========================
# Script execution 
# =========================
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_parquet("../data/processed/data.parquet")

    # Apply preprocessing
    df = preprocess(df)

    # Inspect
    print(df.head())
    print(df.columns)

    # Save processed dataset
    df.to_parquet("../data/processed/data3.parquet", index=False)
    
