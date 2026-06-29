"""Business metric calculations for the e-commerce sales analysis.

All functions are pure: they take prepared DataFrames (as produced by
``data_loader``) and return scalars, Series, or small DataFrames. They perform
no I/O and no plotting, so they can be reused across notebooks, scripts, and
future datasets, and tested in isolation.

Unless noted otherwise, "revenue" means the sum of the item ``price`` column for
the rows passed in. Filter the sales table to the desired status and period with
``data_loader.filter_sales`` before calling these functions.
"""

import numpy as np
import pandas as pd

# Default delivery-speed bucket boundaries (in days). These reproduce the
# original analysis and are exposed as parameters rather than hardcoded so any
# analyst can choose their own boundaries.
DEFAULT_DELIVERY_BINS = (3, 7)
DEFAULT_DELIVERY_LABELS = ("1-3 days", "4-7 days", "8+ days")


def total_revenue(sales):
    """Return total revenue (sum of item ``price``) for the given sales rows."""
    return sales["price"].sum()


def order_count(sales):
    """Return the number of distinct orders in the given sales rows."""
    return sales["order_id"].nunique()


def average_order_value(sales):
    """Return the average order value: mean of per-order revenue totals."""
    return sales.groupby("order_id")["price"].sum().mean()


def monthly_revenue(sales):
    """Return revenue summed by calendar month, indexed 1-12."""
    return sales.groupby("month")["price"].sum()


def monthly_growth(sales):
    """Return month-over-month revenue growth as a fractional Series.

    The first month is ``NaN`` because it has no prior month to compare against.
    """
    return monthly_revenue(sales).pct_change()


def average_monthly_growth(sales):
    """Return the mean month-over-month revenue growth (a fraction)."""
    return monthly_growth(sales).mean()


def growth_rate(current, previous):
    """Return the fractional change from ``previous`` to ``current``.

    Reused for year-over-year revenue, average-order-value, and order-count
    comparisons. Multiply by 100 for a percentage.
    """
    return (current - previous) / previous


def revenue_by_category(sales):
    """Return revenue by product category, sorted from highest to lowest."""
    return (
        sales.groupby("product_category_name")["price"]
        .sum()
        .sort_values(ascending=False)
    )


def revenue_by_state(sales):
    """Return revenue by customer state as a DataFrame, highest first.

    Returns
    -------
    pandas.DataFrame
        Columns ``customer_state`` and ``price`` (revenue), suitable for a
        choropleth map.
    """
    return (
        sales.groupby("customer_state")["price"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def average_review_score(review_table):
    """Return the mean review score in the review/delivery table."""
    return review_table["review_score"].mean()


def review_score_distribution(review_table, normalize=True):
    """Return the distribution of review scores.

    Parameters
    ----------
    review_table : pandas.DataFrame
        Output of ``data_loader.build_review_delivery_table``.
    normalize : bool
        If ``True`` (default) return proportions; otherwise raw counts.

    Returns
    -------
    pandas.Series
        Indexed by review score.
    """
    return review_table["review_score"].value_counts(normalize=normalize)


def review_by_delivery_days(review_table):
    """Return the mean review score for each whole-day delivery duration."""
    return (
        review_table.groupby("delivery_days")["review_score"]
        .mean()
        .reset_index()
    )


def review_by_delivery_bucket(
    review_table, bins=DEFAULT_DELIVERY_BINS, labels=DEFAULT_DELIVERY_LABELS
):
    """Return the mean review score per delivery-speed bucket.

    Parameters
    ----------
    review_table : pandas.DataFrame
        Output of ``data_loader.build_review_delivery_table``.
    bins : sequence of int
        Inclusive upper boundaries (in days) separating the buckets. The default
        ``(3, 7)`` yields the buckets "1-3 days", "4-7 days", and "8+ days".
    labels : sequence of str
        Bucket labels; must contain exactly ``len(bins) + 1`` entries.

    Returns
    -------
    pandas.DataFrame
        Columns ``delivery_time`` (bucket label) and ``review_score`` (mean).
    """
    if len(labels) != len(bins) + 1:
        raise ValueError("labels must have exactly len(bins) + 1 entries")

    edges = [-np.inf, *bins, np.inf]
    bucketed = review_table.copy()
    bucketed["delivery_time"] = pd.cut(
        bucketed["delivery_days"], bins=edges, labels=list(labels)
    )
    return (
        bucketed.groupby("delivery_time", observed=False)["review_score"]
        .mean()
        .reset_index()
    )


def average_delivery_days(review_table):
    """Return the average delivery time in days."""
    return review_table["delivery_days"].mean()


def order_status_distribution(orders, year=None, month=None):
    """Return the proportion of orders by status for a period.

    Operates on the raw orders table (not the line-item sales table) so the
    proportions reflect orders, not order lines.

    Parameters
    ----------
    orders : pandas.DataFrame
        Raw orders dataset.
    year : int, optional
        Restrict to orders purchased in this year. ``None`` keeps all years.
    month : int, optional
        Restrict to orders purchased in this month. ``None`` keeps the full year.

    Returns
    -------
    pandas.Series
        Status proportions, indexed by ``order_status``.
    """
    purchased = pd.to_datetime(orders["order_purchase_timestamp"])
    mask = pd.Series(True, index=orders.index)
    if year is not None:
        mask &= purchased.dt.year == year
    if month is not None:
        mask &= purchased.dt.month == month
    return orders[mask]["order_status"].value_counts(normalize=True)
