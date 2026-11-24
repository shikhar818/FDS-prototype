from pathlib import Path
import numpy as np
import pandas as pd
from tabpfn.model_loading import load_fitted_tabpfn_model
from dotenv import load_dotenv 
import os 

load_dotenv("app/.env")
MODEL_PATH = os.getenv("REVENUE_MODEL")

def load_revenue_model():
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Revenue model not found: {MODEL_PATH}"
        )
    return load_fitted_tabpfn_model(str(MODEL_PATH))


revenue_model = load_revenue_model()

def predict_revenue(df:pd.DataFrame) -> np.ndarray:
    
    predictions = revenue_model.predict(df)
    predictions = np.asarray(predictions).reshape(-1)
    
    if len(predictions) !=len(df):
        raise RuntimeError(
            "Regression prediction count does not match rows"
        )
    
    if not np.isfinite(predictions).all():
        raise ValueError(
            "Regression model returned invalid predictions."
        )

    return predictions


#if __name__ == "__main__":
#    df = pd.read_csv("app/functions/X_test.csv")
#    model_pred  = predict_revenue(df=df)
#    results = df.copy()
#    results["predicted_revenue"] = model_pred
#    print("Predicted total revenue:",results["predicted_revenue"].sum().round(2))