
import pandas as pd
from app.functions.revenue_prediction import predict_revenue
from app.functions.cancellation_prediction import predict_cancellation
import time 
import tracemalloc 


def run_prediction_pipeline(
    booking_df: pd.DataFrame
) -> pd.DataFrame:

    if booking_df.empty:
        raise ValueError("Input CSV contains no bookings.")
    
    
    tracemalloc.start()
    start_time = time.perf_counter()
    # Preserve original booking data
    result_df = booking_df.copy()


    predicted_revenue = predict_revenue(
        booking_df
    )


    predicted_cancellation = predict_cancellation(
        booking_df
    )
    
    inference_time = time.perf_counter() - start_time 
    current_memory,peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 ** 2)

    throughput = (
        len(booking_df) / inference_time
        if inference_time > 0 else 0
    )
    
    performance = {
        "inference_time_seconds": round(inference_time, 6),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "throughput_rows_per_second": round(throughput, 2)
    }
    
    print(performance)


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
    