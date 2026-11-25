import os
from pathlib import Path
from dotenv import load_dotenv
from tabpfn.model_loading import load_fitted_tabpfn_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

load_dotenv("app/.env")
MODEL_PATH = os.getenv("CLASSIFICATION_MODEL")


def load_cancellation_model():
    if not MODEL_PATH:
        raise ValueError("CLASSIFICATION_MODEL not found in .env")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return load_fitted_tabpfn_model(MODEL_PATH)


cancellation_model = load_cancellation_model()

def predict_cancellation(df:pd.DataFrame) -> np.ndarray:
    predictions = cancellation_model.predict(df)
    predictions = np.asarray(predictions).reshape(-1)
    return predictions.astype(int)

# test = pd.read_csv(
#     r"X_test.csv",
#     index_col=0
# )
# y_test = pd.read_csv(r"y_test.csv")
# predictions_test=predict_cancellation(test.head(1000))
# print(classification_report(y_test, predictions_test))