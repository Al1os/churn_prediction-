from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import joblib

app = FastAPI(
    title="Customer Churn Predictor"
)

model = joblib.load(
    "model.pkl"
)

class UserFeatures(BaseModel):

    days_inactive: float
    num_products: int
    total_quantity: int
    avg_qty_per_product: float
    large_order: int
    purchase_month: int
    single_product_order: int

@app.get("/")
def root():
    return {"message": "Customer Churn Predictor API"}

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.get("/features")
def features():

    return {
        "features": [
            "days_inactive",
            "num_products",
            "total_quantity",
            "avg_qty_per_product",
            "large_order",
            "purchase_month",
            "single_product_order"
        ]
    }


@app.post("/predict")
def predict(user: UserFeatures):

    features = np.array([[
        user.days_inactive,
        user.num_products,
        user.total_quantity,
        user.avg_qty_per_product,
        user.large_order,
        user.purchase_month,
        user.single_product_order
    ]])

    prediction = model.predict(
        features
    )[0]

    probability = model.predict_proba(
        features
    )[0][1]

    return {
        "churned": bool(prediction),
        "churn_probability":
            round(float(probability), 3)
    }