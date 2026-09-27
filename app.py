"""Streamlit dashboard for the next-month sales forecast.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sales_forecasting import (
    IncompatibleModelsError,
    forecast_next_month,
    load_artifacts,
    run_training,
)
from sales_forecasting.evaluation import METHOD_NAMES

st.set_page_config(page_title="AI Sales Forecasting", page_icon="📊", layout="wide")

# Validated categorical slots (light, dark): blue for actuals, orange for the
# comparison series; gray is only for de-emphasised marks.
PALETTE = {
    "light": {"series_1": "#2a78d6", "series_2": "#eb6834", "muted": "#898781", "text": "#52514e"},
    "dark": {"series_1": "#3987e5", "series_2": "#d95926", "muted": "#898781", "text": "#c3c2b7"},
}
BAR_SIZE = 16
MONTH_AXIS = alt.Axis(format="%b %Y", labelAngle=0)
MONEY_AXIS = alt.Axis(format="$~s")
CATEGORY_AXIS = alt.Axis(labelLimit=260, labelOverlap=False)


def colors() -> dict[str, str]:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return PALETTE["dark" if theme == "dark" else "light"]


def money(value: float) -> str:
    """Compact currency: $724.2M, $3.11M, $950K."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M" if abs(value) >= 1e8 else f"${value / 1e6:,.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.0f}K"
    return f"${value:,.0f}"


def value_labels(data: pd.DataFrame, x: str, y: alt.Y, color: str) -> alt.Chart:
    """Value at each bar tip, in text ink rather than the series colour."""
    return (
        alt.Chart(data)
        .mark_text(align="left", dx=4, color=color)
        .encode(x=f"{x}:Q", y=y, text=alt.Text(f"{x}:Q", format="$,.3~s"))
    )


def hover_layers(data: pd.DataFrame, x: alt.X, y: alt.Y, tooltip: list) -> list:
    """Crosshair + tooltip for the month nearest the pointer on a line chart."""
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    targets = (
        alt.Chart(data)
        .mark_point(size=400, opacity=0)
        .encode(x=x, y=y, tooltip=tooltip)
        .add_params(nearest)
    )
    rule = (
        alt.Chart(data)
        .mark_rule(strokeWidth=1, color="#898781")
        .encode(x=x)
        .transform_filter(nearest)
    )
    return [targets, rule]


@st.cache_resource(show_spinner="Loading models and forecasting...")
def get_state():
    """Load saved models and forecast; retrain in memory if they can't be used."""
    note = None
    try:
        artifacts = load_artifacts()
    except (FileNotFoundError, IncompatibleModelsError) as exc:
        note = f"Saved models could not be used ({exc}) Fresh models were trained in memory."
        artifacts = run_training()
    forecast = forecast_next_month(artifacts)
    history = artifacts.monthly.merge(
        artifacts.stores[["store_id", "region", "cluster_label"]], on="store_id"
    )
    return artifacts.metadata, forecast, history, note


metadata, forecast_all, history_all, retrain_note = get_state()
c = colors()
forecast_month = forecast_all["forecast_month"].max()
last_month = forecast_all["date"].max()
tier_order = list(metadata["cluster_labels"].values())

# --- Header -------------------------------------------------------------------
st.title("📊 AI-Driven Sales Forecasting")
st.caption(
    f"Next-month revenue forecast for {len(forecast_all)} country-level stores, "
    f"trained on monthly sales from {pd.Timestamp(metadata['data']['first_month']):%b %Y} "
    f"to {last_month:%b %Y}."
)
if retrain_note:
    st.info(retrain_note)

# --- Sidebar filters ------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    regions = sorted(forecast_all["region"].unique())
    picked_regions = st.multiselect("Region", regions, placeholder="All regions")
    picked_tiers = st.multiselect(
        "Sales tier",
        tier_order,
        placeholder="All tiers",
        help="Stores grouped by KMeans on average sales, volatility and peak month.",
    )
    top_n = st.slider("Stores in ranking", 5, 40, 15)

    st.divider()
    st.markdown("**About the data**")
    excluded = metadata["data"]["excluded_incomplete_month"]
    about = (
        f"- History: {pd.Timestamp(metadata['data']['first_month']):%b %Y} to "
        f"{last_month:%b %Y}\n"
        f"- Model: {metadata['model_description']} "
        f"({METHOD_NAMES[metadata['selected_method']]})\n"
    )
    if excluded:
        about += (
            f"- {pd.Timestamp(excluded):%b %Y} is left out: the raw file stops on "
            f"{pd.Timestamp(metadata['data']['raw_last_order_date']):%d %b %Y}, so that "
            "month is incomplete and is forecast instead.\n"
        )
    st.markdown(about)

mask = pd.Series(True, index=forecast_all.index)
if picked_regions:
    mask &= forecast_all["region"].isin(picked_regions)
if picked_tiers:
    mask &= forecast_all["cluster_label"].isin(picked_tiers)
forecast = forecast_all[mask]
history = history_all[history_all["store_id"].isin(forecast["store_id"])]

if forecast.empty:
    st.warning("No stores match the selected filters.")
    st.stop()

# --- KPI row -------------------------------------------------------------------
total_forecast = forecast["forecast_sales"].sum()
total_last = forecast["last_month_sales"].sum()
overall = metadata["metrics"]["overall"]
selected = overall[metadata["selected_method"]]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Forecast month",
    f"{forecast_month:%B %Y}",
    help=f"The first full month after the history, which ends {last_month:%B %Y}.",
    border=True,
)
k2.metric(
    "Total forecast revenue",
    money(total_forecast),
    delta=f"{(total_forecast / total_last - 1) * 100:+.1f}% vs {last_month:%b %Y}"
    if total_last
    else None,
    border=True,
)
k3.metric("Stores", f"{len(forecast)}", help="After sidebar filters.", border=True)
k4.metric(
    "Monthly total error",
    f"{selected['portfolio_mape']:.1%}",
    help=(
        "Mean absolute % error of the all-store monthly total over the "
        f"{metadata['validation']['months']} hold-out months "
        f"({pd.Timestamp(metadata['validation']['start']):%b %Y} to "
        f"{pd.Timestamp(metadata['validation']['end']):%b %Y}), all stores."
    ),
    border=True,
)

tab_overview, tab_store, tab_model, tab_table = st.tabs(
    ["Overview", "Store explorer", "Model performance", "Forecast table"]
)

# --- Overview ------------------------------------------------------------------
with tab_overview:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Monthly revenue, selected stores")
        totals = history.groupby("date", as_index=False)["sales"].sum()
        totals = totals[totals["date"] > last_month - pd.DateOffset(months=36)]
        totals["series"] = "Actual"
        point = pd.DataFrame(
            {"date": [forecast_month], "sales": [total_forecast], "series": ["Forecast"]}
        )
        bridge = pd.concat([totals.tail(1), point]).assign(series="Forecast")
        color = alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Actual", "Forecast"], range=[c["series_1"], c["series_2"]]),
            legend=alt.Legend(title=None, orient="top"),
        )
        x = alt.X("date:T", title=None, axis=MONTH_AXIS)
        yq = alt.Y("sales:Q", title="Revenue", axis=MONEY_AXIS)
        tooltip = [
            alt.Tooltip("date:T", title="Month", format="%b %Y"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("sales:Q", title="Revenue", format="$,.0f"),
        ]
        both = pd.concat([totals, point], ignore_index=True)
        chart = alt.layer(
            alt.Chart(totals).mark_line(strokeWidth=2).encode(x=x, y=yq, color=color),
            alt.Chart(bridge)
            .mark_line(strokeWidth=2, strokeDash=[4, 3])
            .encode(x=x, y=yq, color=color),
            alt.Chart(point)
            .mark_point(filled=True, size=90, opacity=1)
            .encode(x=x, y=yq, color=color),
            *hover_layers(both, x, yq, tooltip),
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"Orange point: forecast for {forecast_month:%B %Y}; the dashed segment joins it "
            "to the last actual month."
        )

    with right:

        def breakdown(column: str, title: str, order=None):
            data = forecast.groupby(column, as_index=False)["forecast_sales"].sum()
            y = alt.Y(f"{column}:N", title=None, sort=order or "-x", axis=CATEGORY_AXIS)
            bars = (
                alt.Chart(data)
                .mark_bar(size=BAR_SIZE, cornerRadiusEnd=4, color=c["series_1"])
                .encode(
                    x=alt.X(
                        "forecast_sales:Q",
                        title=None,
                        axis=MONEY_AXIS,
                        scale=alt.Scale(domainMax=data["forecast_sales"].max() * 1.45),
                    ),
                    y=y,
                    tooltip=[
                        alt.Tooltip(f"{column}:N", title=title),
                        alt.Tooltip("forecast_sales:Q", title="Forecast", format="$,.0f"),
                    ],
                )
            )
            labels = value_labels(data, "forecast_sales", y, c["text"])
            return (bars + labels).properties(height=max(110, 36 * len(data)))

        st.subheader("Forecast by sales tier")
        tiers = [t for t in tier_order if t in set(forecast["cluster_label"])]
        st.altair_chart(breakdown("cluster_label", "Tier", tiers[::-1]), width="stretch")
        st.subheader("Forecast by region")
        st.altair_chart(breakdown("region", "Region"), width="stretch")

    st.subheader(f"Top {min(top_n, len(forecast))} stores by forecast revenue")
    top = forecast.nlargest(top_n, "forecast_sales")
    long = top.melt(
        id_vars=["store_id", "region", "cluster_label", "forecast_lower", "forecast_upper"],
        value_vars=["forecast_sales", "last_month_sales"],
        var_name="series",
        value_name="revenue",
    )
    long["series"] = long["series"].map(
        {"forecast_sales": f"Forecast {forecast_month:%b %Y}", "last_month_sales": "Last month"}
    )
    domain = [f"Forecast {forecast_month:%b %Y}", "Last month"]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=[c["series_1"], c["series_2"]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("store_id:N", title=None, sort=list(top["store_id"]), axis=CATEGORY_AXIS)
    tooltip = [
        alt.Tooltip("store_id:N", title="Store"),
        alt.Tooltip("region:N", title="Region"),
        alt.Tooltip("cluster_label:N", title="Tier"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f"),
    ]
    ranking = alt.layer(
        alt.Chart(long[long["series"] == domain[0]])
        .mark_bar(size=BAR_SIZE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("revenue:Q", title="Revenue", axis=MONEY_AXIS),
            y=y,
            color=color,
            tooltip=tooltip,
        ),
        alt.Chart(long[long["series"] == domain[1]])
        .mark_tick(thickness=3, size=BAR_SIZE + 6)
        .encode(x="revenue:Q", y=y, color=color, tooltip=tooltip),
    ).properties(height=alt.Step(28))
    st.altair_chart(ranking, width="stretch")
    st.caption("Bars: forecast. Orange ticks: last month's actual revenue.")

# --- Store explorer ----------------------------------------------------------------
with tab_store:
    store_names = sorted(forecast["store_id"])
    top_store = forecast.loc[forecast["forecast_sales"].idxmax(), "store_id"]
    store = st.selectbox("Store", store_names, index=store_names.index(top_store))
    row = forecast.set_index("store_id").loc[store]
    series = history[history["store_id"] == store].sort_values("date")
    store_mean = series["sales"].mean()
    st.caption(f"Region: {row['region']} · Sales tier: {row['cluster_label']}")

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "All-time monthly average",
        money(store_mean),
        help=f"Since {series['date'].min():%b %Y}.",
        border=True,
    )
    s2.metric(f"Actual {last_month:%b %Y}", money(row["last_month_sales"]), border=True)
    s3.metric(
        f"Forecast {forecast_month:%b %Y}",
        money(row["forecast_sales"]),
        delta=f"{row['change_pct']:+.1f}% vs last month" if pd.notna(row["change_pct"]) else None,
        border=True,
    )
    coverage = metadata["interval"]["coverage"]
    # Escaped: two "$" signs in one Streamlit string render as LaTeX math.
    st.markdown(
        f"**{coverage:.0%} forecast range:** \\{money(row['forecast_lower'])} to "
        f"\\{money(row['forecast_upper'])}, from hold-out forecast errors for the "
        f"{row['cluster_label']} tier."
    )

    hist = series[series["date"] > last_month - pd.DateOffset(months=48)].assign(series="Actual")
    point = pd.DataFrame(
        {
            "date": [forecast_month],
            "sales": [row["forecast_sales"]],
            "lower": [row["forecast_lower"]],
            "upper": [row["forecast_upper"]],
            "series": ["Forecast"],
        }
    )
    domain = ["Actual", "Forecast", "Store average"]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=[c["series_1"], c["series_2"], c["muted"]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X("date:T", title=None, axis=MONTH_AXIS)
    yq = alt.Y("sales:Q", title="Revenue", axis=MONEY_AXIS)
    avg = pd.DataFrame({"sales": [store_mean], "series": ["Store average"]})
    hover_data = pd.concat([hist, point], ignore_index=True)
    store_chart = alt.layer(
        alt.Chart(avg).mark_rule(strokeWidth=1).encode(y="sales:Q", color=color),
        alt.Chart(hist).mark_line(strokeWidth=2).encode(x=x, y=yq, color=color),
        alt.Chart(point)
        .mark_rule(strokeWidth=2)
        .encode(x=x, y="lower:Q", y2="upper:Q", color=color),
        alt.Chart(point).mark_point(filled=True, size=90, opacity=1).encode(x=x, y=yq, color=color),
        *hover_layers(
            hover_data,
            x,
            yq,
            [
                alt.Tooltip("date:T", title="Month", format="%b %Y"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("sales:Q", title="Revenue", format="$,.0f"),
                alt.Tooltip("lower:Q", title="Range low", format="$,.0f"),
                alt.Tooltip("upper:Q", title="Range high", format="$,.0f"),
            ],
        ),
    ).properties(height=340)
    st.altair_chart(store_chart, width="stretch")
    st.caption(
        f"Last 4 years shown. Vertical orange line: {coverage:.0%} forecast range. "
        "Gray line: the store's all-time monthly average."
    )

# --- Model performance ---------------------------------------------------------------
with tab_model:
    val = metadata["validation"]
    naive = overall["naive_last_month"]
    best_baseline = min(
        (k for k in overall if k not in ("cluster_models", "global_model")),
        key=lambda k: overall[k]["mae"],
    )
    st.markdown(
        f"Every method was scored on the **{val['months']} hold-out months** "
        f"({pd.Timestamp(val['start']):%b %Y} to {pd.Timestamp(val['end']):%b %Y}, "
        f"{val['rows']:,} store-months) after training only on earlier data. The forecast "
        f"uses the **{METHOD_NAMES[metadata['selected_method']]}** approach "
        f"({metadata['selection']})."
    )

    perf = pd.DataFrame(
        [
            {
                "key": key,
                "Method": METHOD_NAMES.get(key, key),
                "MAE": stats["mae"],
                "RMSE": stats["rmse"],
                "WAPE": stats["wape"] * 100,
                "Monthly total error": stats["portfolio_mape"] * 100,
                "vs naive": stats["improvement_vs_naive_pct"],
            }
            for key, stats in overall.items()
        ]
    ).sort_values("MAE")
    perf["role"] = perf["key"].map(
        lambda k: "Used for forecast" if k == metadata["selected_method"] else "Other"
    )
    y = alt.Y("Method:N", title=None, sort=list(perf["Method"]), axis=CATEGORY_AXIS)
    bars = (
        alt.Chart(perf)
        .mark_bar(size=BAR_SIZE, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "MAE:Q",
                title="Mean absolute error per store-month",
                axis=MONEY_AXIS,
                scale=alt.Scale(domainMax=perf["MAE"].max() * 1.2),
            ),
            y=y,
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=["Used for forecast", "Other"], range=[c["series_1"], c["muted"]]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Method:N"),
                alt.Tooltip("MAE:Q", format="$,.0f"),
                alt.Tooltip("RMSE:Q", format="$,.0f"),
                alt.Tooltip("WAPE:Q", format=".1f"),
            ],
        )
    )
    labels = value_labels(perf, "MAE", y, c["text"])
    st.altair_chart((bars + labels).properties(height=36 * len(perf)), width="stretch")
    st.caption("Blue: the method used for the forecast. Lower is better.")

    st.dataframe(
        perf[["Method", "MAE", "RMSE", "WAPE", "Monthly total error", "vs naive"]],
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="$%,.0f"),
            "RMSE": st.column_config.NumberColumn(format="$%,.0f"),
            "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            "Monthly total error": st.column_config.NumberColumn(format="%.1f%%"),
            "vs naive": st.column_config.NumberColumn(
                "MAE vs naive", format="%+.1f%%", help="Positive = lower error than naive."
            ),
        },
    )

    gain = selected["improvement_vs_naive_pct"]
    st.markdown(
        f"""
**How to read this**

- The model's error is **{gain:.1f}% lower than the naive "next month = last month"
  forecast**, and its all-store monthly total is within **{selected["portfolio_mape"]:.1%}**
  on average, which is what matters for revenue, stock and staffing plans.
- It is **on par with the best simple benchmark**
  ({METHOD_NAMES[best_baseline]}: MAE {money(overall[best_baseline]["mae"])} vs
  {money(selected["mae"])}). Store sales in this
  dataset barely depend on previous months (autocorrelation is close to zero), so most of
  the gain over naive comes from forecasting a store's typical level instead of copying
  last month's noise.
- Individual store-months stay hard to predict (WAPE {selected["wape"]:.0%}); use the
  forecast ranges in the store explorer, not single numbers, for store-level decisions.
- Naive last month MAE for reference: {money(naive["mae"])}.
"""
    )

    with st.expander("Results by sales tier"):
        rows = []
        for cl, block in metadata["metrics"]["by_cluster"].items():
            for key in (metadata["selected_method"], "naive_last_month", best_baseline):
                rows.append(
                    {
                        "Tier": metadata["cluster_labels"][cl],
                        "Stores": metadata["cluster_store_counts"][cl],
                        "Method": METHOD_NAMES[key],
                        "MAE": block[key]["mae"],
                        "WAPE": block[key]["wape"] * 100,
                    }
                )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="$%,.0f"),
                "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

# --- Forecast table ---------------------------------------------------------------------
with tab_table:
    table = forecast.sort_values("forecast_sales", ascending=False)
    st.dataframe(
        table,
        hide_index=True,
        column_order=[
            "store_id",
            "region",
            "cluster_label",
            "last_month_sales",
            "forecast_sales",
            "forecast_lower",
            "forecast_upper",
            "change_pct",
        ],
        column_config={
            "store_id": "Store",
            "region": "Region",
            "cluster_label": "Tier",
            "last_month_sales": st.column_config.NumberColumn(
                f"Actual {last_month:%b %Y}", format="$%,.0f"
            ),
            "forecast_sales": st.column_config.NumberColumn(
                f"Forecast {forecast_month:%b %Y}", format="$%,.0f"
            ),
            "forecast_lower": st.column_config.NumberColumn("Range low", format="$%,.0f"),
            "forecast_upper": st.column_config.NumberColumn("Range high", format="$%,.0f"),
            "change_pct": st.column_config.NumberColumn("Change", format="%+.1f%%"),
        },
    )
    st.download_button(
        "Download forecast CSV",
        forecast.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
        file_name=f"sales_forecast_{forecast_month:%Y_%m}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
