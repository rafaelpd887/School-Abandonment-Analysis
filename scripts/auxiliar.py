import pandas as pd


df_reduced.columns

df_reduced['rede'].value_counts()

df = df[(df['localizacao'] != 'total') & (df['rede'] != 'total')]

df.dtypes

print(y.describe())

df.corr(numeric_only=True)['taxa_abandono_em']\
  .sort_values(ascending=False)
  
  print(X.columns)
  print(X.dtypes)

