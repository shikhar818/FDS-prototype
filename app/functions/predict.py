from fastapi import APIRouter,HTTPException 
from tabpfn.model_loading import load_fitted_tabpfn_model  
import pandas as pd 
import joblib
from typing import Any
from pydantic import BaseModel 





# --------------------------------------------------
# 2. Define JSON request structure
# --------------------------------------------------

class BookingRequest(BaseModel):
    booking: dict[str, Any]


# --------------------------------------------------
# 3. Prepare incoming JSON
# --------------------------------------------------

def prepare_booking(booking: dict) -> pd.DataFrame:
    
    metadata = joblib.load("app/metadata/api_metadata.pkl")
    FEATURE_COLUMNS = metadata["features"]
    CATEGORICAL_FEATURES = metadata["categorical_features"]
    CATEGORY_LEVELS = metadata["category_level"]


    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in booking
    ]

    extra_features = [
        col for col in booking
        if col not in FEATURE_COLUMNS
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    # Maintain the exact training feature order
    input_df = pd.DataFrame(
        [booking],
        columns=FEATURE_COLUMNS
    )

    # Restore categorical feature types
    for col in CATEGORICAL_FEATURES:

        value = input_df[col].iloc[0]

        if pd.isna(value):
            raise ValueError(
                f"Categorical feature '{col}' cannot be empty."
            )

        allowed_categories = CATEGORY_LEVELS[col]

        if value not in allowed_categories:
            raise ValueError(
                f"Invalid value '{value}' for '{col}'. "
                f"Use a category observed during training."
            )

        input_df[col] = pd.Categorical(
            input_df[col],
            categories=allowed_categories
        )

    # Convert remaining features to numeric
    numerical_features = [
        col for col in FEATURE_COLUMNS
        if col not in CATEGORICAL_FEATURES
    ]

    for col in numerical_features:
        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="raise"
        )

    if input_df.isnull().any().any():
        raise ValueError(
            "Booking contains missing feature values."
        )

    return input_df
