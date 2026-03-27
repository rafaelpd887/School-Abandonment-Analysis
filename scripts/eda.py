import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load data
# =========================
df = pd.read_pickle("data/processed/data.pkl")

# =========================
# 2. Quick overview
# =========================
print("Dataset shape:", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget summary:\n", df['taxa_abandono_em'].describe())

# =========================
# 3. Correlation with target
# =========================
corr = df.corr(numeric_only=True)['taxa_abandono_em'].sort_values(ascending=False)
print("\nCorrelation with target:\n", corr)

# =========================
# 4. Pairplot of top correlated features
# =========================
top_features = corr.index[1:6].tolist()  # top 5 features excluding target
sns.pairplot(df[top_features + ['taxa_abandono_em']])
plt.suptitle("Pairplot of top correlated features", y=1.02)
plt.show()

# =========================
# 5. Distribution of target
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df['taxa_abandono_em'], bins=20, kde=True)
plt.title("Distribution of school abandonment rates")
plt.xlabel("taxa_abandono_em")
plt.ylabel("Frequency")
plt.show()

# =========================
# 6. Boxplots for categorical variables
# =========================
categorical_features = ['localizacao', 'rede']
for cat in categorical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=cat, y='taxa_abandono_em', data=df)
    plt.title(f"Target vs {cat}")
    plt.show()

