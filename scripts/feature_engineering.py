import pandas as pd

# Load the dataset
df = pd.read_pickle("data/processed/data.pkl")

# Drop highly correlated columns
cols_drop = ['tdi_em', 'taxa_aprovacao_em']
df = df.drop(columns=cols_drop)

# =========================
# Feature Engineering
# =========================

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

# =========================
# Inspect the result
# =========================
print(df.head())
print(df.columns)

# =========================
# Save the dataset ready for modeling
# =========================
df.to_pickle("data/processed/data2.pkl")
