import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_pickle("data/processed/data2.pkl")

# =========================
# 1. Split features and target
# =========================
TARGET = "taxa_abandono_em"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# 2. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 3. Define categorical features
# =========================
categorical_features = ['localizacao', 'rede']  

# =========================
# 4. Create LightGBM datasets
# =========================
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_features
)

test_data = lgb.Dataset(
    X_test,
    label=y_test,
    categorical_feature=categorical_features,
    reference=train_data
)

# =========================
# 5. Model parameters (REGRESSION)
# =========================
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 12,
    "max_depth": 4,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

# =========================
# 6. Train model
# =========================
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# =========================
# 7. Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. Evaluation
# =========================
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
