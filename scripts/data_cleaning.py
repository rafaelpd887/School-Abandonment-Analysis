import pandas as pd

#LOAD DATA
df = pd.read_csv(r"..\data\raw\br_inep_indicadores_educacionais_brasil.csv.gz", compression="gzip")                 

#verifying variables
#columns = list(df.columns)
#print(columns)

# List of reduced predictor columns
predictors = [
    'localizacao',        # school location
    'rede',               # school network/type
    'atu_em',             # average students per class - High School
    'had_em',             # average daily teaching hours - High School
    'tdi_em',             # age-grade distortion rate - High School
    'taxa_aprovacao_em',  # historical pass rate - High School
    'taxa_reprovacao_em', # historical fail rate - High School
    'dsu_em',             # percentage of teachers with higher education - High School
    'afd_em_grupo_1'      # teachers with most suitable training (group 1)
]
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

# Add the target column
columns_final = predictors + ['taxa_abandono_em']

# Create the reduced DataFrame
df_reduced = df[columns_final].copy()

# Clean NaN from the target
df_reduced = df_reduced.dropna(subset=['taxa_abandono_em']).copy()

# Adjusting categories for LightGBM
df_reduced['localizacao'] = df_reduced['localizacao'].astype('category')
df_reduced['rede'] = df_reduced['rede'].astype('category')

# Saving
df_reduced.to_pickle("../data/processed/data.pkl")