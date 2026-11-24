
import pandas as pd
from revenue_prediction import predict_revenue
from cancellation_prediction import predict_cancellation
import time 


def run_prediction_pipeline(
    booking_df: pd.DataFrame
) -> pd.DataFrame:

    if booking_df.empty:
        raise ValueError("Input CSV contains no bookings.")

    # Preserve original booking data
    result_df = booking_df.copy()


    predicted_revenue = predict_revenue(
        booking_df
    )


    predicted_cancellation = predict_cancellation(
        booking_df
    )


    result_df["predicted_revenue"] = predicted_revenue

    result_df["predicted_cancellation"] = (
        predicted_cancellation
    )

    result_df["cancellation_status"] = (
        result_df["predicted_cancellation"]
        .map({
            0: "Not Canceled",
            1: "Canceled"
        })
    )

    return result_df


#if __name__=="__main__":
#    start = time.perf_counter()
#    df = pd.read_csv("app/functions/X_test.csv")
#    print(run_prediction_pipeline(df))
#    end = time.perf_counter()
#    time_taken = end - start 
#    print(time_taken)
    