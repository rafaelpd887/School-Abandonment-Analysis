from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np

# =========================
# 1 Evaluate on Test Set
# =========================
# Compute RMSE and R2 on your held-out test set
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")  # Root Mean Squared Error
print(f"R2: {r2:.4f}")      # Coefficient of Determination

# =========================
# 2 Cross-Validation
# =========================
# Create a LightGBM regressor for CV
model_cv = lgb.LGBMRegressor(**params)

# 5-fold cross-validation using negative RMSE
scores = cross_val_score(
    model_cv,
    X,
    y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)

rmse_cv = -scores.mean()  # Convert negative RMSE to positive
print(f"CV RMSE: {rmse_cv:.4f}")  # RMSE from cross-validation

# =========================
# 3 Inspect Target Distribution
# =========================
# Quick overview of the target variable
print(y.describe())

# =========================
# 4 Feature Correlations
# =========================
# Check correlation of features with target to identify strong predictors
print(df.corr(numeric_only=True)['taxa_abandono_em'].sort_values(ascending=False))

