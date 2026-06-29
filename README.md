# e-dash

E-commerce sales exploratory data analysis. The analysis answers business
questions about revenue, products, geography, and customer experience from the
order data in `ecommerce_data/`.

## Project structure

- `ecommerce_data/` - raw CSV datasets (orders, order items, products,
  customers, reviews, payments).
- `data_loader.py` - data layer: loads the CSVs, builds a single analysis-ready
  sales table, and filters it by period.
- `business_metrics.py` - pure metric calculations (revenue, average order
  value, growth, category and state revenue, review and delivery metrics).
- `EDA_Refactored.ipynb` - the documented, configurable analysis notebook.
- `EDA.ipynb` - the original exploratory notebook, kept for reference.
- `app.py` - interactive Streamlit sales dashboard built on the modules above.
- `requirements.txt` - Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Running the analysis

Launch Jupyter and open the refactored notebook, then run all cells:

```bash
jupyter notebook EDA_Refactored.ipynb
```

Or execute it non-interactively:

```bash
jupyter nbconvert --to notebook --execute EDA_Refactored.ipynb
```

## Configuring the analysis period

The analysis period is set in the notebook's Configuration section. Change these
variables and re-run all cells to analyze any period:

- `ANALYSIS_YEAR` - the primary year to analyze (default `2023`).
- `COMPARISON_YEAR` - the year to compare against (default `2022`).
- `ANALYSIS_MONTH` - a specific month `1-12` to restrict to, or `None` for the
  full year (default `None`).

The metric functions in `business_metrics.py` are general purpose: they operate
on whatever filtered sales table they are given, so the same code works for any
year, month, or future dataset.

## Dashboard

`app.py` is an interactive Streamlit dashboard that reuses `data_loader.py` and
`business_metrics.py`. Run it from the project root:

```bash
streamlit run app.py
```

Layout:

- Header with the dashboard title and a global date-range filter.
- KPI row: Total Revenue, Monthly Growth, Average Order Value, Total Orders.
- A 2x2 chart grid: revenue trend (current vs previous period), top 10
  categories by revenue, revenue by state (US choropleth), and average review
  score by delivery-time bucket.
- Bottom row: average delivery time and average review score.

The date-range filter drives every KPI and chart. Each metric is compared
against the equal-length period immediately preceding the selected range; for a
full calendar year this is the prior year, matching the notebook analysis.
Upward and downward trends are shown in green and red respectively, with the
delivery-time card treating faster delivery as the positive direction. All
charts are rendered with Plotly.
