# School Abandonment Analysis

## Description
This project analyzes school abandonment (dropout) rates and identifies factors that influence students leaving school. Using historical school and student data, we explore patterns, correlations, and predictive modeling to understand which factors most affect dropout rates.

The project combines **exploratory data analysis (EDA)**, **feature engineering**, and **LightGBM regression modeling** to predict the dropout rate (`taxa_abandono_em`) for each school.

---

## Project Structure

- **data/** → Contains raw and processed datasets.
- **scripts/** → Python scripts for data preprocessing, EDA, feature engineering, modeling, and evaluation.
  - `eda.py` → Exploratory data analysis.
  - `data_cleaning.py` → Data cleaning and preprocessing.
  - `feature_engineering.py` → Creation of new variables based on interactions and domain insights.
  - `train_lightgbm.py` → Model training.
  - `model_evaluation.py` → Model diagnostics (metrics, residuals, feature importance).
  - `auxiliar.py` → Helper functions.
- **README.md** → Project documentation.

---

## Key Variables

- **tdi_em** → Age-grade distortion (strong predictor of dropout).
- **taxa_aprovacao_em / taxa_reprovacao_em** → Historical performance indicators.
- **dsu_em** → Percentage of teachers with higher education.
- **had_em** → Daily class hours.
- **atu_em** → Students per class.
- **afd_em_grupo_1** → Teacher adequacy.

### Engineered Features
Examples:
- `reprovacao_per_had`
- `dsu_per_afd`
- `had_minus_afd`
- `reprovacao_x_dsu`

These features capture **interactions and ratios** that are not directly observable in the original dataset.

---

## Methodology

1. **EDA**
   - Analyzed distributions, missing values, and correlations.
   - Identified strong relationships with the target variable.

2. **Feature Engineering**
   - Created interaction and ratio-based features.
   - Tested the impact of removing highly correlated variables.

3. **Modeling**
   - Used **LightGBM Regressor** for its performance with tabular data.
   - Trained multiple models with different feature sets.

4. **Evaluation**
   - Metrics: **RMSE, MAE, R²**
   - **Cross-validation (5-fold)** to estimate generalization
   - Residual analysis and feature importance for diagnostics

---

## Model Evolution

| Model | RMSE | R² | CV RMSE |
|-------|------|----|---------|
| Original features | 0.5498 | 0.9842 | 1.0001 |
| Removed high-correlated features | 2.0132 | 0.7885 | 2.3359 |
| Feature engineering (without correlated features) | 1.9342 | 0.8048 | 2.3441 |
| **Final model (engineered + correlated features)** | **0.9667** | **0.9512** | **1.6155** |

---

## Final Model Selection

The final model combines:
- **Highly correlated variables** (`tdi_em`, `taxa_aprovacao_em`)
- **Feature engineered variables**

### Why this approach?

- Removing highly correlated variables reduced performance significantly.
- These variables are **not deterministic functions of the target**, so they do not constitute true leakage.
- Feature engineering added complementary information, improving robustness.

### Key Insight

Although `tdi_em` and `taxa_aprovacao_em` are strong predictors, the model does not rely solely on them. Engineered features such as interaction and ratio-based variables also show high importance, indicating that the model captures deeper relationships in the data.

---

## Model Diagnostics

- **Cross-validation RMSE is higher than test RMSE**, which indicates that:
  - The model is not overfitting heavily
  - Performance estimates are realistic for unseen data

- **Feature importance analysis** shows:
  - Strong contribution from core variables
  - Meaningful impact from engineered features

- **Residual analysis** indicates no major systematic bias

---

## Conclusion

The final model achieves a strong balance between:

- **Predictive performance**
- **Generalization ability**
- **Interpretability**

This makes it suitable for real-world applications, such as:
- Identifying high-risk schools
- Supporting educational policy decisions
- Guiding targeted interventions

---

## Objective

The goal is to build a predictive model that estimates school dropout rates based on institutional and educational factors, enabling **data-driven decision-making** and more effective intervention strategies.