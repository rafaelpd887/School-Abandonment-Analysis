from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Evaluation on Test Set
# =========================
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=== Test Set Performance ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")


# =========================
# 2. Cross-Validation
# =========================
model_cv = lgb.LGBMRegressor(**params)

scores = cross_val_score(
    model_cv,
    X,
    y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)

rmse_cv = -scores.mean()

print("\n=== Cross-Validation ===")
print(f"CV RMSE: {rmse_cv:.4f}")


# =========================
# 3. Target Distribution
# =========================
print("\n=== Target Distribution ===")
print(y.describe())


# =========================
# 4. Feature Correlation
# =========================
print("\n=== Feature Correlation with Target ===")
print(df.corr(numeric_only=True)['taxa_abandono_em']
      .sort_values(ascending=False))


# =========================
# 5. Feature Importance
# =========================
print("\n=== Plotting Feature Importance ===")

lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# =========================
# 6. Residual Analysis
# =========================
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()


# =========================
# 7. Actual vs Predicted
# =========================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()


# =========================
# 8. Residual Distribution
# =========================
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()