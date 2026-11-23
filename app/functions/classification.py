from tabpfn.model_loading import load_fitted_tabpfn_model
import os
import json
import pandas as pd 
from fastapi import HTTPException

model_path="app/models/trained_reg.tabpfn_fit"
model = load_fitted_tabpfn_model(model_path)

with open("app/models/feature_columns.json") as f:
    features = json.load(f)

# booking=

def preprocess(booking):
    df=pd.DataFrame([booking])
    df=pd.get_dummies(df)
    df=df.reindex(columns=features,fill_value=0)
    return df

def predict(df):
    prediction = model.predict(df)
    return int(prediction[0])

def classify_booking(booking: dict):
    try:
        processed = preprocess(booking)
        prediction = predict(processed)
        return {
            "prediction": prediction,
            "result": "Canceled" if prediction == 1 else "Not Canceled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))