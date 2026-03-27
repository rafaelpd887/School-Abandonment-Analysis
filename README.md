# School Abandonment Analysis

## Description
This project analyzes school abandonment (dropout) rates and identifies factors that influence students leaving school. Using historical school and student data, we explore patterns, correlations, and predictive modeling to understand which factors most affect dropout rates.

The project combines **exploratory data analysis (EDA)**, **feature engineering**, and **LightGBM regression modeling** to predict the dropout rate (`taxa_abandono_em`) for each school.

## Project Structure
The repository is organized as follows:

- **data/** → Contains raw and processed datasets.
- **scripts/** → Python scripts for data preprocessing, EDA, feature engineering, modeling, and evaluation.
  - `eda.py` → Simplified exploratory data analysis.
  - `data_cleaning.py` → Data cleaning and preprocessing (removing NA, type conversions, etc.).
  - `feature_engineering.py` → Creates new features based on domain knowledge and feature interactions.
  - `train_lightgbm.py` → Trains LightGBM regression models.
  - `model_evaluation.py` → Evaluates models with metrics like RMSE, R², and cross-validation.
  - `auxiliar.py` → Helper functions (e.g., inspect variable names, types, basic summaries).
- **README.md** → Project documentation and results.

## Key Variables
- **localizacao** → School location; captures regional differences in dropout rates.
- **rede** → School network (public, private, or other); impacts resources and dropout.
- **atu_em** → Average number of students per class; large classes may increase dropout.
- **had_em** → Daily class hours; extreme workloads can affect dropout risk.
- **tdi_em** → Age-grade distortion; more students out of the correct grade increases risk.
- **taxa_aprovacao_em / taxa_reprovacao_em** → Historical approval and failure rates; included carefully to avoid target leakage.
- **dsu_em** → Percentage of teachers with higher education; more qualified teachers can reduce dropout.
- **afd_em_grupo_1** → Teacher adequacy (best-trained group); higher quality teaching reduces abandonment risk.
- **Engineered features** → e.g., `reprovacao_per_had`, `dsu_per_afd`, `had_plus_afd`; created based on interactions and ratios between existing features.

## Workflow
1. Work in the `dev` branch.
2. Add, commit, and push your changes to `dev`.
3. Merge tested and stable changes from `dev` into `main`.

## Methodology
1. **Exploratory Data Analysis (EDA)** → Understand distributions, missing values, and correlations among variables.
2. **Feature Engineering** → Remove highly correlated features and create new features to better capture patterns in the data.
3. **Modeling** → Train LightGBM regression models to predict school dropout rate (`taxa_abandono_em`) using both original and engineered features.
4. **Evaluation** → Compare models using RMSE, R², and cross-validation RMSE to select the best model for production.

## Results

| Model | RMSE | R² | CV RMSE |
|-------|------|----|---------|
| Original features | 0.5498 | 0.9842 | 1.0001 |
| Removed high-correlated features | 2.0132 | 0.7885 | 2.3359 |
| Feature engineered | 1.9342 | 0.8048 | 2.3441 |

**Best model for production:** Feature engineered model

## Objective
The goal is to build a predictive model that estimates the probability of students dropping out based on school and student characteristics, enabling better intervention strategies and data-driven policy decisions.