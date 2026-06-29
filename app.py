"""Streamlit sales dashboard for the e-commerce data.

Reuses the analysis modules:
- ``data_loader``: load CSVs, build the sales table, join reviews/delivery.
- ``business_metrics``: pure metric calculations.

A single date-range filter in the header drives every KPI and chart. Each metric
is compared against the equal-length period immediately preceding the selection
(for a full year, that is the prior year, matching the original notebook).
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import business_metrics as bm
import data_loader

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
PRIMARY_BLUE = "#1f4e79"
PREVIOUS_GREY = "#9aa0a6"
POSITIVE_GREEN = "#1a9850"
NEGATIVE_RED = "#d73027"
STAR_GOLD = "#f5a623"
GRID_COLOR = "#ececec"
BLUE_SCALE = "Blues"

st.set_page_config(page_title="E-commerce Sales Dashboard", layout="wide")

CARD_CSS = """
<style>
.block-container { padding-top: 3.5rem; }
.dash-card {
    background: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-card { height: 132px; }
.info-card { height: 150px; }
.card-label {
    font-size: 0.80rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.card-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.card-trend { font-size: 0.95rem; font-weight: 600; margin-top: 8px; }
.stars { color: %(star)s; font-size: 1.5rem; letter-spacing: 3px; margin: 4px 0; }
.dash-title { font-size: 1.9rem; font-weight: 700; color: #111827; line-height: 1.3; margin: 0; padding-top: 4px; }
</style>
""" % {"star": STAR_GOLD}

st.markdown(CARD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load raw data once and build the analysis-ready sales table."""
    datasets = data_loader.load_raw_datasets()
    sales = data_loader.build_sales_dataset(
        datasets["orders"],
        datasets["order_items"],
        datasets["products"],
        datasets["customers"],
    )
    return sales, datasets["reviews"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _trim(number):
    """Format a float with up to two decimals, dropping trailing zeros."""
    return f"{number:.2f}".rstrip("0").rstrip(".")


def format_dollars(value):
    """Compact currency: 300000 -> $300K, 2000000 -> $2M, 724.98 -> $724.98."""
    if value is None or pd.isna(value):
        return "--"
    amount = abs(float(value))
    sign = "-" if value < 0 else ""
    if amount >= 1_000_000:
        return f"{sign}${_trim(amount / 1_000_000)}M"
    if amount >= 1_000:
        return f"{sign}${_trim(amount / 1_000)}K"
    return f"{sign}${amount:,.2f}"


def format_percent(value):
    """Format a fraction as a percentage with two decimals."""
    if value is None or pd.isna(value):
        return "--"
    return f"{value * 100:.2f}%"


def dollar_axis(max_value, n_ticks=5):
    """Return (tickvals, ticktext) of evenly spaced compact-dollar ticks."""
    if max_value is None or pd.isna(max_value) or max_value <= 0:
        return [0], ["$0"]
    raw_step = max_value / n_ticks
    magnitude = 10 ** int(np.floor(np.log10(raw_step)))
    step = np.ceil(raw_step / magnitude) * magnitude
    ticks = np.arange(0, max_value + step, step)
    return list(ticks), [format_dollars(t) for t in ticks]


def safe_growth(current, previous):
    """Fractional change, or None when it cannot be computed."""
    if previous in (None, 0) or pd.isna(previous) or pd.isna(current):
        return None
    return (current - previous) / previous


def trend_html(growth, invert=False):
    """Build a colored trend indicator. ``invert`` treats a decrease as good."""
    if growth is None or pd.isna(growth):
        return '<div class="card-trend" style="color:#9aa0a6">--</div>'
    going_up = growth >= 0
    is_good = (not going_up) if invert else going_up
    color = POSITIVE_GREEN if is_good else NEGATIVE_RED
    arrow = "▲" if going_up else "▼"
    return (
        f'<div class="card-trend" style="color:{color}">'
        f"{arrow} {abs(growth) * 100:.2f}%</div>"
    )


def star_rating(score, max_stars=5):
    """Filled/empty stars for a 0-5 score, rounded to the nearest whole star."""
    if score is None or pd.isna(score):
        return "☆" * max_stars
    filled = int(round(score))
    filled = max(0, min(max_stars, filled))
    return "★" * filled + "☆" * (max_stars - filled)


def kpi_card(label, value, trend=""):
    return (
        f'<div class="dash-card kpi-card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>{trend}</div>'
    )


def monthly_revenue_series(sales):
    """Revenue summed per calendar month, indexed by monthly Period."""
    if sales.empty:
        return pd.Series(dtype=float)
    grouped = sales.copy()
    grouped["period"] = grouped["order_purchase_timestamp"].dt.to_period("M")
    return grouped.groupby("period")["price"].sum().sort_index()


def slice_sales(sales, start, end, status=data_loader.DELIVERED_STATUS):
    """Delivered sales with a purchase date within [start, end] (inclusive)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    purchased = sales["order_purchase_timestamp"]
    mask = (
        (sales["order_status"] == status)
        & (purchased >= start_ts)
        & (purchased < end_ts)
    )
    return sales[mask].copy()


def base_chart_layout(fig, title):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.01, font=dict(size=16)),
        height=360,
        margin=dict(l=10, r=10, t=46, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
sales, reviews = load_data()

delivered = sales[sales["order_status"] == data_loader.DELIVERED_STATUS]
min_date = delivered["order_purchase_timestamp"].min().date()
max_date = delivered["order_purchase_timestamp"].max().date()

# Default to the most recent year with substantial data (ignoring sparse tail
# years), so the dashboard opens on a meaningful full-year view.
year_counts = delivered["order_purchase_timestamp"].dt.year.value_counts()
substantial_years = year_counts[year_counts >= 0.1 * year_counts.max()].index
default_year = int(max(substantial_years))
default_start = max(pd.Timestamp(year=default_year, month=1, day=1).date(), min_date)
default_end = min(pd.Timestamp(year=default_year, month=12, day=31).date(), max_date)

# Header: title (left) + global date-range filter (right).
header_left, header_right = st.columns([3, 2])
with header_left:
    st.markdown('<p class="dash-title">E-commerce Sales Dashboard</p>', unsafe_allow_html=True)
with header_right:
    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
    )

if not (isinstance(date_range, (list, tuple)) and len(date_range) == 2):
    st.info("Select a start and end date to view the dashboard.")
    st.stop()

start, end = date_range
period_length = end - start
previous_end = start - timedelta(days=1)
previous_start = previous_end - period_length

current = slice_sales(sales, start, end)
previous = slice_sales(sales, previous_start, previous_end)

if current.empty:
    st.warning("No delivered orders in the selected date range.")
    st.stop()

st.caption(
    f"Current period: {start:%b %d, %Y} to {end:%b %d, %Y}  |  "
    f"Compared with: {previous_start:%b %d, %Y} to {previous_end:%b %d, %Y}"
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
current_revenue = bm.total_revenue(current)
previous_revenue = bm.total_revenue(previous)
current_aov = bm.average_order_value(current)
previous_aov = bm.average_order_value(previous)
current_orders = bm.order_count(current)
previous_orders = bm.order_count(previous)

current_monthly = monthly_revenue_series(current)
previous_monthly = monthly_revenue_series(previous)
monthly_growth_avg = current_monthly.pct_change().mean()

current_reviews = data_loader.build_review_delivery_table(current, reviews)
previous_reviews = data_loader.build_review_delivery_table(previous, reviews)
current_delivery = bm.average_delivery_days(current_reviews)
previous_delivery = bm.average_delivery_days(previous_reviews)
current_review_score = bm.average_review_score(current_reviews)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.markdown(
    kpi_card("Total Revenue", format_dollars(current_revenue),
             trend_html(safe_growth(current_revenue, previous_revenue))),
    unsafe_allow_html=True,
)
kpi2.markdown(
    kpi_card("Monthly Growth", format_percent(monthly_growth_avg)),
    unsafe_allow_html=True,
)
kpi3.markdown(
    kpi_card("Average Order Value", format_dollars(current_aov),
             trend_html(safe_growth(current_aov, previous_aov))),
    unsafe_allow_html=True,
)
kpi4.markdown(
    kpi_card("Total Orders", f"{current_orders:,}",
             trend_html(safe_growth(current_orders, previous_orders))),
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------------------
# Charts grid (2 x 2)
# ---------------------------------------------------------------------------
row1_left, row1_right = st.columns(2)

# Revenue trend: solid current vs dashed previous, aligned month-by-month.
with row1_left:
    x_labels = [p.strftime("%b %Y") for p in current_monthly.index]
    revenue_fig = go.Figure()
    revenue_fig.add_trace(
        go.Scatter(
            x=x_labels, y=current_monthly.values, mode="lines+markers",
            name="Current period", line=dict(color=PRIMARY_BLUE, width=3),
        )
    )
    if not previous_monthly.empty:
        n = min(len(x_labels), len(previous_monthly))
        revenue_fig.add_trace(
            go.Scatter(
                x=x_labels[:n], y=previous_monthly.values[:n], mode="lines+markers",
                name="Previous period",
                line=dict(color=PREVIOUS_GREY, width=2, dash="dash"),
            )
        )
    max_rev = max(
        current_monthly.max(),
        previous_monthly.max() if not previous_monthly.empty else 0,
    )
    tickvals, ticktext = dollar_axis(max_rev)
    revenue_fig.update_yaxes(tickvals=tickvals, ticktext=ticktext,
                             showgrid=True, gridcolor=GRID_COLOR)
    revenue_fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    base_chart_layout(revenue_fig, "Revenue Trend")
    revenue_fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(revenue_fig, width="stretch")

# Top 10 categories: blue gradient (light = lower value), values as $K/$M.
with row1_right:
    top_categories = bm.revenue_by_category(current).head(10)
    cats = list(top_categories.index)[::-1]
    vals = list(top_categories.values)[::-1]
    category_fig = go.Figure(
        go.Bar(
            x=vals, y=cats, orientation="h",
            marker=dict(color=vals, colorscale=BLUE_SCALE, showscale=False),
            text=[format_dollars(v) for v in vals],
            textposition="outside", cliponaxis=False,
        )
    )
    tickvals, ticktext = dollar_axis(max(vals) if vals else 0)
    category_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    base_chart_layout(category_fig, "Top 10 Categories by Revenue")
    st.plotly_chart(category_fig, width="stretch")

row2_left, row2_right = st.columns(2)

# Revenue by state: US choropleth with blue gradient.
with row2_left:
    state_revenue = bm.revenue_by_state(current)
    state_fig = px.choropleth(
        state_revenue, locations="customer_state", color="price",
        locationmode="USA-states", scope="usa", color_continuous_scale=BLUE_SCALE,
        labels={"price": "Revenue", "customer_state": "State"},
    )
    state_fig.update_coloraxes(colorbar_tickprefix="$", colorbar_tickformat="~s")
    base_chart_layout(state_fig, "Revenue by State")
    state_fig.update_geos(bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(state_fig, width="stretch")

# Satisfaction vs delivery time.
with row2_right:
    bucket_df = bm.review_by_delivery_bucket(current_reviews)
    satisfaction_fig = go.Figure(
        go.Bar(
            x=bucket_df["delivery_time"].astype(str),
            y=bucket_df["review_score"], marker_color=PRIMARY_BLUE,
            text=[f"{v:.2f}" if pd.notna(v) else "" for v in bucket_df["review_score"]],
            textposition="outside", cliponaxis=False,
        )
    )
    satisfaction_fig.update_yaxes(range=[0, 5], title="Average review score")
    satisfaction_fig.update_xaxes(title="Delivery time")
    base_chart_layout(satisfaction_fig, "Satisfaction vs Delivery Time")
    st.plotly_chart(satisfaction_fig, width="stretch")

st.write("")

# ---------------------------------------------------------------------------
# Bottom row
# ---------------------------------------------------------------------------
bottom_left, bottom_right = st.columns(2)

delivery_value = "--" if pd.isna(current_delivery) else f"{current_delivery:.2f} days"
bottom_left.markdown(
    f'<div class="dash-card info-card"><div class="card-label">Average Delivery Time</div>'
    f'<div class="card-value">{delivery_value}</div>'
    f"{trend_html(safe_growth(current_delivery, previous_delivery), invert=True)}</div>",
    unsafe_allow_html=True,
)

review_value = "--" if pd.isna(current_review_score) else f"{current_review_score:.2f}"
bottom_right.markdown(
    f'<div class="dash-card info-card">'
    f'<div class="card-value" style="font-size:2.4rem">{review_value}</div>'
    f'<div class="stars">{star_rating(current_review_score)}</div>'
    f'<div class="card-label" style="margin-top:6px">Average Review Score</div></div>',
    unsafe_allow_html=True,
)
