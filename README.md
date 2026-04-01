# School Abandonment Analysis

🔗 **Live App:** https://school-dropout-prediction.streamlit.app/

---

## Description

This project analyzes school dropout rates and identifies the main factors that influence students leaving school.

It combines **exploratory data analysis (EDA)**, **feature engineering**, and a **LightGBM regression model** to predict the dropout rate (`taxa_abandono_em`) for each school.

Additionally, the project includes a **live deployed dashboard**, allowing users to explore data and generate predictions interactively.

---

## 🚀 Live Demo

Access the deployed dashboard:

👉 https://school-dropout-prediction.streamlit.app/

### Features:
- Interactive filters (school network, location)
- Real-time dropout predictions
- Distribution visualization
- Ranking of highest-risk schools
- Downloadable filtered dataset

---

## 🏗️ Deployment & Architecture

- Model trained using **LightGBM**
- Model serialized with `joblib`
- Interactive dashboard built with **Streamlit**
- Deployed on **Streamlit Community Cloud**

### Pipeline Structure:
- Data cleaning → `scripts/data_cleaning.py`
- Feature engineering → `scripts/feature_engineering.py`
- Model training → `scripts/train_lightgbm.py`
- Inference integrated into the dashboard

This project simulates a real-world ML workflow:


---

## Project Structure

INEP/

│

├── dashboard/

│ └── app.py

│

├── scripts/

│ ├── __init__.py

│ ├── eda.py

│ ├── data_cleaning.py

│ ├── feature_engineering.py

│ ├── train_lightgbm.py

│ ├── model_evaluation.py

│ └── auxiliar.py

│

├── models/

│ └── model.pkl

│

├── data/

│ └── processed/

│ └── data.csv

│

├── requirements.txt

└── README.md


---

## Key Variables

- **tdi_em** → Age-grade distortion (strong predictor of dropout)
- **taxa_aprovacao_em / taxa_reprovacao_em** → Academic performance indicators
- **dsu_em** → Percentage of teachers with higher education
- **had_em** → Daily class hours
- **atu_em** → Students per class
- **afd_em_grupo_1** → Teacher adequacy

### Engineered Features

Examples:
- `reprovacao_per_had`
- `dsu_per_afd`
- `had_minus_afd`
- `reprovacao_x_dsu`

These features capture interactions and ratios that are not explicitly available in the original dataset.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed distributions and missing values
- Identified correlations with dropout rate
- Detected key predictive variables

### 2. Feature Engineering
- Created interaction-based and ratio-based features
- Tested the impact of removing highly correlated variables

### 3. Modeling
- Algorithm: **LightGBM Regressor**
- Optimized for tabular data performance
- Tested multiple feature configurations

### 4. Evaluation
- Metrics: **RMSE, MAE, R²**
- **5-fold cross-validation**
- Residual analysis
- Feature importance analysis

---

## Model Evolution

| Model | RMSE | R² | CV RMSE |
|------|------|----|---------|
| Original features | 0.5498 | 0.9842 | 1.0001 |
| Removed correlated features | 2.0132 | 0.7885 | 2.3359 |
| Engineered (without correlated) | 1.9342 | 0.8048 | 2.3441 |
| **Final model** | **0.9667** | **0.9512** | **1.6155** |

---

## Final Model Selection

The final model combines:
- Highly correlated variables (`tdi_em`, `taxa_aprovacao_em`)
- Feature engineered variables

### Why?

- Removing correlated variables significantly reduced performance
- These variables are not deterministic → no true leakage
- Feature engineering adds complementary information

---

## Model Diagnostics

- Cross-validation RMSE higher than test RMSE → realistic generalization
- Feature importance shows:
  - Strong base variables
  - Relevant engineered features
- Residual analysis shows no major bias

---

## Conclusion

The model achieves a strong balance between:

- Predictive performance
- Generalization
- Interpretability

### Practical Applications:
- Identifying high-risk schools
- Supporting public policy decisions
- Targeting educational interventions

---

## ⚙️ How to Run Locally

```bash
git clone <your-repo>
cd <your-repo>

pip install -r requirements.txt

streamlit run dashboard/app.py
```
---

## Objective

To build an end-to-end machine learning solution capable of predicting school dropout rates and enabling data-driven decision-making through an interactive and deployed application.