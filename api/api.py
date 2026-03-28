from fastapi import FastAPI
import joblib
import pandas as pd

# Import preprocessing function (must match training pipeline)
from scripts.feature_engineering import preprocess

# Initialize FastAPI app
app = FastAPI(
    title="School Abandonment Prediction API",
    version="1.0",
    description="""
Predicts school dropout rates using a LightGBM model.

---

### 📥 Expected Input (JSON)

Use the following structure when sending a request:

```json
{
  "localizacao": "Urbana",
  "rede": "Estadual",
  "atu_em": 30,
  "had_em": 5,
  "tdi_em": 10,
  "taxa_aprovacao_em": 80,
  "taxa_reprovacao_em": 10,
  "dsu_em": 70,
  "afd_em_grupo_1": 60
}
"""
)
    
# Load trained model
# Adjust path if necessary depending on your project structure
model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    """
    Health check endpoint.
    Used to verify if the API is running correctly.
    """
    return {"message": "School Abandonment API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    Prediction endpoint.

    Parameters:
    - data (dict): Input data in JSON format representing one observation.

    Process:
    1. Convert input JSON into a pandas DataFrame
    2. Apply preprocessing (feature engineering)
    3. Generate prediction using trained model

    Returns:
    - prediction (float): Predicted dropout rate
    """
    try:
        # Convert input JSON into DataFrame
        df = pd.DataFrame([data])

        # Apply same preprocessing used during training
        X = preprocess(df)

        # Generate prediction
        prediction = model.predict(X)

        # Return result as JSON
        return {
            "prediction": float(prediction[0])
        }

    except Exception as e:
        # Return error message for debugging
        return {"error": str(e)}