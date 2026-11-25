import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict/batch"

# Palette: retained = teal, canceled/loss = brick red, neutral = slate
RETAINED = "#1f7a6d"
CANCELED = "#c0392b"
NEUTRAL = "#51606e"
GRID = "rgba(128,128,128,0.2)"

st.set_page_config(page_title="Hotel cancellations and revenue", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1300px;}
    div[data-testid="stMetric"] {
        border-left: 3px solid #51606e; padding: 0.2rem 0 0.2rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def money(x: float) -> str:
    return f"{x:,.0f}"


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def call_api(file) -> dict:
    files = {"file": (file.name, file.getvalue(), "text/csv")}
    r = requests.post(API_URL, files=files, timeout=1200)
    r.raise_for_status()
    return r.json()


# ---------- sidebar ----------
with st.sidebar:
    st.header("Upload bookings")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    run = st.button("Run predictions", type="primary", disabled=uploaded is None)
    st.caption(f"Endpoint: {API_URL}")

if run and uploaded is not None:
    with st.spinner("Running predictions..."):
        try:
            st.session_state["result"] = call_api(uploaded)
            st.session_state["source"] = uploaded.name
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Check that FastAPI is running on 127.0.0.1:8000.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned {e.response.status_code}: {e.response.text[:500]}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Request failed: {e}")

result = st.session_state.get("result")

st.title("Hotel cancellations and revenue loss")

if not result:
    st.info("Upload a CSV in the sidebar and click Run predictions.")
    st.stop()

if result.get("status") != "success":
    st.error(f"API reported status: {result.get('status')}")
    st.json(result)
    st.stop()

s = result["summary"]
charts = result["charts"]
st.caption(f"Source: {st.session_state.get('source')}  |  Predictions file: {result.get('filename')}")

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total bookings", f"{s['total_bookings']:,}")
c2.metric("Predicted cancellations", f"{s['canceled_bookings']:,}", f"{s['cancellation_rate']}% of bookings", delta_color="off")
c3.metric("Revenue at risk", money(s["lost_revenue"]), f"{s['revenue_loss_percentage']}% of total", delta_color="off")
c4.metric("Avg booking value", f"{s['average_booking_value']:,.2f}")

c5, c6, _, _ = st.columns(4)
c5.metric("Total potential revenue", money(s["total_revenue"]))
c6.metric("Retained revenue", money(s["retained_revenue"]))

tab_overview, tab_segments, tab_monthly = st.tabs(["Overview", "Channels and segments", "Monthly trend"])

# ---------- Overview ----------
with tab_overview:
    left, right = st.columns(2)

    with left:
        df = pd.DataFrame(charts["booking_outcomes"])
        fig = go.Figure(
            go.Pie(
                labels=df["status"], values=df["bookings"], hole=0.55,
                marker=dict(colors=[RETAINED if x == "Retained" else CANCELED for x in df["status"]]),
                textinfo="label+percent",
            )
        )
        fig.update_layout(title="Booking outcomes")
        st.plotly_chart(style(fig), use_container_width=True)

    with right:
        df = pd.DataFrame(charts["revenue_distribution"])
        fig = go.Figure(
            go.Pie(
                labels=df["status"], values=df["revenue"], hole=0.55,
                marker=dict(colors=[RETAINED if "Retained" in x else CANCELED for x in df["status"]]),
                textinfo="label+percent",
            )
        )
        fig.update_layout(title="Revenue distribution")
        st.plotly_chart(style(fig), use_container_width=True)

    left, right = st.columns(2)

    with left:
        wf = pd.DataFrame(charts["revenue_waterfall"])
        fig = go.Figure(
            go.Waterfall(
                x=wf["category"], y=wf["value"], measure=wf["measure"],
                increasing=dict(marker=dict(color=NEUTRAL)),
                decreasing=dict(marker=dict(color=CANCELED)),
                totals=dict(marker=dict(color=RETAINED)),
                connector=dict(line=dict(color=GRID)),
                text=[money(v) for v in wf["value"]], textposition="outside",
            )
        )
        fig.update_layout(title="Revenue waterfall", showlegend=False)
        st.plotly_chart(style(fig), use_container_width=True)

    with right:
        avg = pd.DataFrame(charts["average_booking_value"])
        fig = go.Figure(
            go.Bar(
                x=avg["cancellation_status"], y=avg["predicted_revenue"],
                marker_color=[CANCELED if x == "Canceled" else RETAINED for x in avg["cancellation_status"]],
                text=[f"{v:,.2f}" for v in avg["predicted_revenue"]], textposition="outside",
            )
        )
        fig.update_layout(title="Average booking value by outcome", showlegend=False)
        st.plotly_chart(style(fig), use_container_width=True)

    st.subheader("Revenue by outcome")
    cs = pd.DataFrame(charts["cancellation_summary"]).rename(
        columns={
            "cancellation_status": "Outcome", "total_bookings": "Bookings",
            "total_revenue": "Total revenue", "average_revenue": "Mean",
            "median_revenue": "Median", "minimum_revenue": "Min", "maximum_revenue": "Max",
        }
    )
    st.dataframe(
        cs.style.format({c: "{:,.2f}" for c in cs.columns if c not in ("Outcome", "Bookings")} | {"Bookings": "{:,}"}),
        use_container_width=True, hide_index=True,
    )


# ---------- Channels and segments ----------
def breakdown(rows: list, key: str, label: str, prefix: str = ""):
    df = pd.DataFrame(rows)
    df[key] = prefix + df[key].astype(str)
    df = df.sort_values("total_bookings", ascending=False)

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_bar(x=df[key], y=df["retained_revenue"], name="Retained", marker_color=RETAINED)
        fig.add_bar(x=df[key], y=df["lost_revenue"], name="Lost", marker_color=CANCELED)
        fig.update_layout(barmode="stack", title=f"Revenue by {label}")
        st.plotly_chart(style(fig), use_container_width=True)

    with b:
        fig = go.Figure()
        fig.add_bar(x=df[key], y=df["cancellation_rate"], name="Cancellation rate %", marker_color=NEUTRAL)
        fig.add_bar(x=df[key], y=df["revenue_loss_pct"], name="Revenue loss %", marker_color=CANCELED)
        fig.update_layout(barmode="group", title=f"Cancellation rate vs revenue loss by {label}")
        st.plotly_chart(style(fig), use_container_width=True)

    fig = go.Figure()
    fig.add_bar(x=df[key], y=df["retained_bookings"], name="Retained", marker_color=RETAINED)
    fig.add_bar(x=df[key], y=df["canceled_bookings"], name="Canceled", marker_color=CANCELED)
    fig.update_layout(barmode="stack", title=f"Bookings by {label}")
    st.plotly_chart(style(fig, 300), use_container_width=True)

    show = df.rename(
        columns={
            key: label.title(), "total_bookings": "Bookings", "canceled_bookings": "Canceled",
            "retained_bookings": "Retained", "total_revenue": "Total revenue",
            "lost_revenue": "Lost revenue", "retained_revenue": "Retained revenue",
            "average_booking_value": "Avg value", "cancellation_rate": "Cancel %",
            "revenue_loss_pct": "Revenue loss %",
        }
    )
    st.dataframe(
        show.style.format(
            {
                "Bookings": "{:,}", "Canceled": "{:,}", "Retained": "{:,}",
                "Total revenue": "{:,.0f}", "Lost revenue": "{:,.0f}", "Retained revenue": "{:,.0f}",
                "Avg value": "{:,.2f}", "Cancel %": "{:.2f}", "Revenue loss %": "{:.2f}",
            }
        ),
        use_container_width=True, hide_index=True,
    )


with tab_segments:
    st.subheader("Distribution channel")
    breakdown(charts["distribution_channel"], "distributionchannel", "channel")
    st.divider()
    st.subheader("Market segment")
    st.caption("Segments are numeric codes in the API output; map them to names if you have the encoding.")
    breakdown(charts["market_segment"], "marketsegment", "segment", prefix="Segment ")

# ---------- Monthly trend ----------
with tab_monthly:
    monthly = pd.DataFrame(charts.get("monthly_revenue", []))
    if monthly.empty:
        st.info("The API response has no charts.monthly_revenue key yet. Add it on the backend.")
    else:
        monthly["period"] = pd.to_datetime(
            dict(year=monthly["arrivaldateyear"], month=monthly["arrivalmonth"], day=1)
        )
        monthly = monthly.sort_values("period")

        peak_rev = monthly.loc[monthly["retained_revenue"].idxmax()]
        peak_can = monthly.loc[monthly["cancellation_rate"].idxmax()]
        m1, m2 = st.columns(2)
        m1.metric("Peak revenue month", f"{peak_rev['period']:%b %Y}", money(peak_rev["retained_revenue"]), delta_color="off")
        m2.metric("Peak cancellation month", f"{peak_can['period']:%b %Y}", f"{peak_can['cancellation_rate']:.2f}%", delta_color="off")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4],
            subplot_titles=("Predicted booking revenue by arrival month", "Cancellation rate by arrival month"),
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["period"], y=monthly["retained_revenue"], mode="lines+markers",
                name="Expected revenue", line=dict(color=RETAINED, width=2.5),
                hovertemplate="%{x|%b %Y}<br>Expected revenue: %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
        if "lost_revenue" in monthly.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly["period"], y=monthly["lost_revenue"], mode="lines+markers",
                    name="Revenue lost to cancellations", line=dict(color=CANCELED, width=2, dash="dot"),
                    hovertemplate="%{x|%b %Y}<br>Lost revenue: %{y:,.0f}<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=monthly["period"], y=monthly["cancellation_rate"], mode="lines+markers",
                name="Cancellation rate %", line=dict(color=CANCELED, width=2.5),
                hovertemplate="%{x|%b %Y}<br>Cancellation rate: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.update_xaxes(tickformat="%b %Y", dtick="M1", tickangle=-45)
        fig.update_yaxes(title_text="Revenue", row=1, col=1)
        fig.update_yaxes(title_text="Cancellation rate (%)", row=2, col=1)
        st.plotly_chart(style(fig, 620), use_container_width=True)

        with st.expander("Monthly data"):
            st.dataframe(
                monthly.drop(columns=["period"]),
                use_container_width=True, hide_index=True,
            )

st.sidebar.divider()
st.sidebar.download_button(
    "Download raw JSON",
    data=json.dumps(result, indent=2),
    file_name="predictions_response.json",
    mime="application/json",
)