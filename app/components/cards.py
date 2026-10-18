"""KPI cards for the dashboard."""
import streamlit as st


def format_money(value: float, currency: str = "EUR") -> str:
    return f"{value:,.2f} {currency}"


def kpi_row(items) -> None:
    """Show a row of metric cards. items = [(label, value, delta_or_None), ...]"""
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta)