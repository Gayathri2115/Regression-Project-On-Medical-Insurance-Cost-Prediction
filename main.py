# main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# 1. Create FastAPI App
# =========================
app = FastAPI(title="Medical Insurance Cost Prediction API")

# =========================
# 2. Load Trained Model
# =========================
model = joblib.load("model.pkl")


# =========================
# 3. Define Input Schema
# =========================
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# =========================
# 4. Home Route
# =========================
@app.get("/")
def home():
    return {"message": "Medical Insurance Cost Prediction API is running!"}


# =========================
# 5. Prediction Route
# =========================
@app.post("/predict")
def predict(data: InsuranceInput):

    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "predicted_insurance_cost": round(float(prediction), 2)
    }
