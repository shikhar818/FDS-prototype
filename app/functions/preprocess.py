import pandas as pd

def prepare_classification(
    df: pd.DataFrame,
    predicted_revenue
) -> pd.DataFrame:

    input_df = df.copy()
    input_df["revenue"] = predicted_revenue
    return input_df