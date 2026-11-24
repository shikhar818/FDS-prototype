import pandas as pd
import json

def records(df):
    return json.loads((df.round(2).to_json(orient="records")))

def evaluate_segments(df,group_col):
    summary=df.groupby(group_col).agg(
        total_bookings=("prediction_cancellation","count"),
        canceled_bookings=("prediction_cancellation","sum"),
        total_revenue=("predicted_revenue","sum"),
        lost_revenue=('retained_revenue',"sum"),
        retained_revenue=("retained_revenue","sum"),
        average_booking_value=("predicted_revenue", "mean")
    ).reset_index()

    summary["retained_bookings"]=(summary["total_bookings"]- summary["canceled_bookings"])
    summary["cancellation_rate"]=(summary["canceled_bookings"]- summary["total_bookings"])
    summary["revenue_loss_pct"]=(summary["lost_revenue"]- summary["total_revenue"].replace(0, float("nan"))*100)
    return records(summary)
