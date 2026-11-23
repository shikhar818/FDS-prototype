from fastapi import APIRouter,HTTPException 
from tabpfn.model_loading import load_fitted_tabpfn_model  
import pandas as pd 
import joblib
from typing import Any
from pydantic import BaseModel 
from app.functions.predict import BookingRequest,prepare_booking


router = APIRouter()
model_path = 'app/models/tabpfn_single_regressor.tabpfn_fit'
model = load_fitted_tabpfn_model(model_path)

# --------------------------------------------------
# 4. API home endpoint
# --------------------------------------------------

@router.get("/health_check")
def home():
    return {
        "message": "Hotel Revenue Prediction API is running",
        "model": "TabPFN Regressor",
        "prediction_endpoint": "/predict"
    }


# --------------------------------------------------
# 5. Revenue prediction endpoint
# --------------------------------------------------

@router.post("/predict")
def predict_revenue(request: BookingRequest):

    try:
        input_df = prepare_booking(request.booking)

        prediction = model.predict(input_df)

        predicted_revenue = float(prediction[0])

        return {
            "success": True,
            "predicted_revenue": round(predicted_revenue, 2),
            "currency": "Euro"
        }

    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
