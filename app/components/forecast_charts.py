"""Plotly charts for the Forecast page."""
import pandas as pd
import plotly.graph_objects as go

NAVY, TEAL, GOLD = "#0B2545", "#13A89E", "#F2B134"
PALETTE = ["#0B2545", "#13A89E", "#F2B134", "#7B5EA7", "#D64550", "#9AA0AE",
           "#3E7CB1", "#8FC93A", "#E07A5F", "#5F6C37"]


def forecast_fan_chart(history: pd.DataFrame, fc) -> go.Figure:
    """Last 12 months of income and spending, plus the forecast with its 80% range."""
    last = history.tail(12)
    fig = go.Figure()
    for column, name, color, values in (
        ("income", "Income", TEAL, fc.income),
        ("expenses", "Spending", NAVY, fc.expenses),
    ):
        fig.add_trace(go.Scatter(x=last["month"], y=last[column], name=name,
                                 mode="lines+markers", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(
            x=[last["month"].iloc[-1], fc.target_month],
            y=[last[column].iloc[-1], values["forecast"]],
            mode="lines", line=dict(color=color, dash="dash"), showlegend=False,
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[fc.target_month], y=[values["forecast"]], mode="markers",
            name=f"{name} forecast",
            marker=dict(color=GOLD, size=11, symbol="diamond", line=dict(color=color, width=2)),
            error_y=dict(type="data", symmetric=False, color=color,
                         array=[values["upper"] - values["forecast"]],
                         arrayminus=[values["forecast"] - values["lower"]])))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=380,
                      legend=dict(orientation="h", y=-0.2),
                      yaxis_title=None, xaxis_title=None,
                      title="Income and spending, with the forecast range")
    return fig


def forecast_donut(table: pd.DataFrame) -> go.Figure:
    """Category split of the forecast spending."""
    parts = table[~table["series"].isin(["Income", "Total expenses"]) & (table["forecast"] > 0.5)]
    fig = go.Figure(go.Pie(labels=parts["series"], values=parts["forecast"], hole=0.62,
                           marker=dict(colors=PALETTE), textinfo="percent", sort=True))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=380, title="Where the money goes",
        annotations=[dict(text=f"{parts['forecast'].sum():,.0f}<br>forecast spend",
                          x=0.5, y=0.5, showarrow=False)])
    return fig

def anomaly_scatter(tx: pd.DataFrame, flags: pd.DataFrame) -> go.Figure:
    """All transactions as grey dots; flagged ones coloured by review status."""
    data = tx.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["size"] = data["amount"].abs()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["size"], mode="markers", name="Transactions",
        marker=dict(color="lightgrey", size=5), hoverinfo="skip"))

    colours = {"pending": "orange", "confirmed": "crimson", "dismissed": "steelblue"}
    status = flags.set_index("id")["status"]
    flagged = data[data["id"].isin(status.index)].copy()
    flagged["status"] = flagged["id"].map(status)
    for name, colour in colours.items():
        part = flagged[flagged["status"] == name]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["date"], y=part["size"], mode="markers", name=name.capitalize(),
            marker=dict(color=colour, size=10),
            text=part["description"],
            hovertemplate="%{text}<br>%{x|%d.%m.%Y}: %{y:,.2f}<extra></extra>"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Amount (absolute)", legend_title_text="")
    return fig
