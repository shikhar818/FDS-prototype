import pandas as pd
import json

def records(df):
    return json.loads((df.round(2).to_json(orient="records")))

def evaluate_segments(df,group_col):
    summary=df.groupby(group_col).agg(
        total_bookings=("predicted_cancellation","count"),
        canceled_bookings=("predicted_cancellation","sum"),
        total_revenue=("predicted_revenue","sum"),
        lost_revenue=('revenue_loss',"sum"),
        retained_revenue=("retained_revenue","sum"),
        average_booking_value=("predicted_revenue", "mean")
    ).reset_index()

    summary["retained_bookings"]=(summary["total_bookings"]-summary["canceled_bookings"])
    summary["cancellation_rate"]=(summary["canceled_bookings"]/summary["total_bookings"]*100)
    summary["revenue_loss_pct"]=(summary["lost_revenue"]/summary["total_revenue"].replace(0, float("nan"))*100)
    return records(summary)

def generate_dashboard(df):

    df = df.copy()

    # Booking metrics
    total_bookings = len(df)

    canceled_bookings = int(
        df["predicted_cancellation"].sum()
    )

    retained_bookings = total_bookings - canceled_bookings

    cancellation_rate = (
        canceled_bookings / total_bookings * 100
    )

    # Revenue calculations
    df["revenue_loss"] = (
        df["predicted_revenue"]
        * df["predicted_cancellation"]
    )

    df["retained_revenue"] = (
        df["predicted_revenue"]
        * (1 - df["predicted_cancellation"])
    )

    total_revenue = float(df["predicted_revenue"].sum())
    lost_revenue = float(df["revenue_loss"].sum())
    retained_revenue = float(df["retained_revenue"].sum())

    revenue_loss_percentage = (
        lost_revenue / total_revenue * 100
        if total_revenue != 0 else 0
    )

    average_booking_value = float(
        df["predicted_revenue"].mean()
    )

    df["cancellation_status"] = df[
        "predicted_cancellation"
    ].map({
        0: "Retained",
        1: "Canceled"
    })

    # Cancellation summary
    cancellation_summary = df.groupby(
        "cancellation_status"
    )["predicted_revenue"].agg(
        ["count", "sum", "mean", "median", "min", "max"]
    ).reset_index()

    cancellation_summary.columns = [
        "cancellation_status",
        "total_bookings",
        "total_revenue",
        "average_revenue",
        "median_revenue",
        "minimum_revenue",
        "maximum_revenue"
    ]

    # Average booking value
    average_revenue = df.groupby(
        "cancellation_status"
    )["predicted_revenue"].mean().reset_index()

    # Top 10 high-value bookings
    high_value_bookings = df.sort_values(
        "predicted_revenue",
        ascending=False
    ).head(10)

    return {
        "summary": {
            "total_bookings": total_bookings,
            "canceled_bookings": canceled_bookings,
            "retained_bookings": retained_bookings,
            "cancellation_rate": round(cancellation_rate, 2),
            "total_revenue": round(total_revenue, 2),
            "lost_revenue": round(lost_revenue, 2),
            "retained_revenue": round(retained_revenue, 2),
            "revenue_loss_percentage": round(
                revenue_loss_percentage, 2
            ),
            "average_booking_value": round(
                average_booking_value, 2
            )
        },

        "charts": {
            "booking_outcomes": [
                {
                    "status": "Retained",
                    "bookings": retained_bookings
                },
                {
                    "status": "Canceled",
                    "bookings": canceled_bookings
                }
            ],

            "revenue_distribution": [
                {
                    "status": "Retained Revenue",
                    "revenue": round(retained_revenue, 2)
                },
                {
                    "status": "Potential Revenue Loss",
                    "revenue": round(lost_revenue, 2)
                }
            ],

            "revenue_waterfall": [
                {
                    "category": "Potential Revenue",
                    "measure": "absolute",
                    "value": round(total_revenue, 2)
                },
                {
                    "category": "Cancellation Loss",
                    "measure": "relative",
                    "value": round(-lost_revenue, 2)
                },
                {
                    "category": "Retained Revenue",
                    "measure": "total",
                    "value": round(retained_revenue, 2)
                }
            ],

            "cancellation_summary": records(
                cancellation_summary
            ),

            "average_booking_value": records(
                average_revenue
            ),

            "distribution_channel": evaluate_segments(
                df, "distributionchannel"
            ),

            "market_segment": evaluate_segments(
                df, "marketsegment"
            )
        },

        "high_value_bookings": records(
            high_value_bookings
        ),
    }
    
## Test Code Success 
#if __name__=="__main__":
#    df = pd.read_csv("app/functions/predictions.csv")
#    dashboard_data = generate_dashboard(df)
#    print(dashboard_data)