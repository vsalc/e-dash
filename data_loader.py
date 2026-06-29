"""Data loading, cleaning, and preparation for the e-commerce sales analysis.

This module owns the data layer only: reading the raw CSV files, merging them
into a single analysis-ready sales table, and filtering that table by period.
It performs no metric calculations and no plotting.

The core analysis grain is one row per order line item. Reviews are deliberately
not merged into the core sales table, because an inner join on reviews would drop
orders that have no review and corrupt revenue and order-count metrics. Reviews
are joined separately for the customer-experience analysis via
``build_review_delivery_table``.
"""

import os

import pandas as pd

DATA_DIR = "ecommerce_data"
DELIVERED_STATUS = "delivered"

# Raw CSV files keyed by the name used throughout the analysis.
_DATASET_FILES = {
    "orders": "orders_dataset.csv",
    "order_items": "order_items_dataset.csv",
    "products": "products_dataset.csv",
    "customers": "customers_dataset.csv",
    "reviews": "order_reviews_dataset.csv",
    "payments": "order_payments_dataset.csv",
}


def load_raw_datasets(data_dir=DATA_DIR):
    """Load all raw CSV datasets from ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw ``*_dataset.csv`` files.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of dataset name (orders, order_items, products, customers,
        reviews, payments) to its DataFrame.
    """
    return {
        name: pd.read_csv(os.path.join(data_dir, filename))
        for name, filename in _DATASET_FILES.items()
    }


def build_sales_dataset(orders, order_items, products, customers):
    """Merge raw tables into an analysis-ready sales table.

    The result has one row per order line item, enriched with order status,
    parsed timestamps, derived period columns, delivery duration, product
    category, and customer state.

    Parameters
    ----------
    orders, order_items, products, customers : pandas.DataFrame
        Raw datasets as returned by :func:`load_raw_datasets`.

    Returns
    -------
    pandas.DataFrame
        Columns: order_id, order_item_id, product_id, price, order_status,
        order_purchase_timestamp, order_delivered_customer_date, year, month,
        delivery_days, product_category_name, customer_state.
    """
    sales = pd.merge(
        order_items[["order_id", "order_item_id", "product_id", "price"]],
        orders[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
            ]
        ],
        on="order_id",
    ).copy()

    # Parse timestamps once, up front, so every consumer works with datetimes.
    sales["order_purchase_timestamp"] = pd.to_datetime(
        sales["order_purchase_timestamp"]
    )
    sales["order_delivered_customer_date"] = pd.to_datetime(
        sales["order_delivered_customer_date"]
    )

    # Derived period and duration columns via the .dt accessor (vectorized).
    sales["year"] = sales["order_purchase_timestamp"].dt.year
    sales["month"] = sales["order_purchase_timestamp"].dt.month
    sales["delivery_days"] = (
        sales["order_delivered_customer_date"] - sales["order_purchase_timestamp"]
    ).dt.days

    # Enrich with product category and customer state.
    sales = sales.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    sales = sales.merge(
        customers[["customer_id", "customer_state"]], on="customer_id", how="left"
    )

    return sales


def filter_sales(sales, year=None, month=None, status=DELIVERED_STATUS):
    """Filter the sales table by order status, year, and optional month.

    Parameters
    ----------
    sales : pandas.DataFrame
        Sales table from :func:`build_sales_dataset`.
    year : int, optional
        Keep only orders purchased in this year. ``None`` keeps all years.
    month : int, optional
        Keep only orders purchased in this month (1-12). ``None`` keeps the
        full year.
    status : str, optional
        Order status to keep. ``None`` keeps all statuses. Defaults to
        ``"delivered"``.

    Returns
    -------
    pandas.DataFrame
        A filtered copy of ``sales``.
    """
    mask = pd.Series(True, index=sales.index)
    if status is not None:
        mask &= sales["order_status"] == status
    if year is not None:
        mask &= sales["year"] == year
    if month is not None:
        mask &= sales["month"] == month
    return sales[mask].copy()


def build_review_delivery_table(sales_filtered, reviews):
    """Join reviews onto filtered sales for the customer-experience analysis.

    Reviews are inner-joined to the filtered sales table and deduplicated to one
    row per order, since a review is recorded at the order level while sales are
    at the line-item level.

    Parameters
    ----------
    sales_filtered : pandas.DataFrame
        Output of :func:`filter_sales`; must include ``order_id`` and
        ``delivery_days``.
    reviews : pandas.DataFrame
        Raw reviews dataset (needs ``order_id`` and ``review_score``).

    Returns
    -------
    pandas.DataFrame
        One row per order with columns: order_id, delivery_days, review_score.
    """
    merged = sales_filtered.merge(
        reviews[["order_id", "review_score"]], on="order_id"
    )
    return merged[["order_id", "delivery_days", "review_score"]].drop_duplicates()
