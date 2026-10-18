"""Plotly chart builders."""
import pandas as pd
import plotly.graph_objects as go

INCOME_COLOR = "#12a89d"
SPEND_COLOR = "#0b2545"


def income_vs_spending_chart(monthly: pd.DataFrame) -> go.Figure:
    """Line chart of monthly income and spending."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["income"], mode="lines+markers",
            name="Income", line=dict(color=INCOME_COLOR, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["expenses"], mode="lines+markers",
            name="Spending", line=dict(color=SPEND_COLOR, width=3),
        )
    )
    fig.update_layout(
        title="Income vs spending",
        yaxis_title="EUR",
        xaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def category_donut(transactions: pd.DataFrame) -> go.Figure:
    """Donut chart of spending by category (transactions without a category are grouped)."""
    fig = go.Figure()
    expenses = transactions[transactions["amount"] < 0].copy()
    if expenses.empty:
        fig.update_layout(title="Spending by category (no expenses yet)")
        return fig

    if "category" not in expenses.columns:
        expenses["category"] = None
    expenses["category"] = expenses["category"].fillna("Uncategorized")
    totals = (-expenses.groupby("category")["amount"].sum()).sort_values(ascending=False)

    fig.add_trace(
        go.Pie(labels=list(totals.index), values=list(totals.values), hole=0.55, sort=False)
    )
    fig.update_layout(
        title="Spending by category",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig