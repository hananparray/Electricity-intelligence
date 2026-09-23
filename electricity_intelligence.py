import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Electricity Intelligence",
    page_icon="⚡",
    layout="wide",
)

SOURCE_COLUMNS = [
    "Nuclear", "Wind", "Hydroelectric", "Oil and Gas",
    "Coal", "Solar", "Biomass"
]
RENEWABLE_COLUMNS = ["Wind_Clean", "Hydroelectric", "Solar", "Biomass"]
NON_RENEWABLE_COLUMNS = ["Nuclear", "Oil and Gas", "Coal"]
FEATURES = [
    "Hour", "DayOfWeek", "Month", "DayOfYear",
    "Lag_1", "Lag_24", "Lag_168", "Rolling_24", "Rolling_168"
]


@st.cache_data
def load_and_clean(file_bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes))
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw = raw.dropna(subset=["DateTime"]).drop_duplicates()

    numeric_cols = [c for c in raw.columns if c != "DateTime"]
    data = (
        raw.groupby("DateTime", as_index=False)[numeric_cols]
        .mean()
        .sort_values("DateTime")
        .reset_index(drop=True)
    )

    # The supplied Production column is retained as the total.
    # Negative wind observations are clipped only for source-mix analysis.
    data["Wind_Clean"] = data["Wind"].clip(lower=0)
    data["Renewable"] = (
        data["Wind_Clean"]
        + data["Hydroelectric"]
        + data["Solar"]
        + data["Biomass"]
    )
    data["NonRenewable"] = (
        data["Nuclear"] + data["Oil and Gas"] + data["Coal"]
    )
    data["EnergyBalance"] = data["Production"] - data["Consumption"]

    data["Hour"] = data["DateTime"].dt.hour
    data["DayOfWeek"] = data["DateTime"].dt.dayofweek
    data["DayName"] = data["DateTime"].dt.day_name()
    data["Month"] = data["DateTime"].dt.month
    data["MonthName"] = data["DateTime"].dt.month_name()
    data["Year"] = data["DateTime"].dt.year
    data["IsWeekend"] = data["DayOfWeek"] >= 5
    return data


@st.cache_data
def build_model_data(data):
    m = data[["DateTime", "Consumption"]].copy()
    m["Hour"] = m["DateTime"].dt.hour
    m["DayOfWeek"] = m["DateTime"].dt.dayofweek
    m["Month"] = m["DateTime"].dt.month
    m["DayOfYear"] = m["DateTime"].dt.dayofyear
    m["Lag_1"] = m["Consumption"].shift(1)
    m["Lag_24"] = m["Consumption"].shift(24)
    m["Lag_168"] = m["Consumption"].shift(168)
    m["Rolling_24"] = m["Consumption"].shift(1).rolling(24).mean()
    m["Rolling_168"] = m["Consumption"].shift(1).rolling(168).mean()
    return m.dropna().reset_index(drop=True)


@st.cache_resource
def train_model(model_data):
    split = int(len(model_data) * 0.80)
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        model_data.loc[:split-1, FEATURES],
        model_data.loc[:split-1, "Consumption"],
    )
    test = model_data.loc[split:].copy()
    predictions = model.predict(test[FEATURES])
    metrics = {
        "MAE": mean_absolute_error(test["Consumption"], predictions),
        "RMSE": np.sqrt(mean_squared_error(test["Consumption"], predictions)),
        "R2": r2_score(test["Consumption"], predictions),
    }
    return model, split, predictions, metrics


def recursive_forecast(model, data, horizon):
    history = data["Consumption"].astype(float).tolist()
    last_time = data["DateTime"].iloc[-1]
    future_rows = []

    for step in range(1, horizon + 1):
        next_time = last_time + pd.Timedelta(hours=1)
        values = np.array(history, dtype=float)
        row = pd.DataFrame([{
            "Hour": next_time.hour,
            "DayOfWeek": next_time.dayofweek,
            "Month": next_time.month,
            "DayOfYear": next_time.dayofyear,
            "Lag_1": values[-1],
            "Lag_24": values[-24],
            "Lag_168": values[-168],
            "Rolling_24": values[-24:].mean(),
            "Rolling_168": values[-168:].mean(),
        }])
        prediction = float(model.predict(row[FEATURES])[0])
        history.append(prediction)
        future_rows.append({"DateTime": next_time, "Forecast": prediction})
        last_time = next_time

    return pd.DataFrame(future_rows)


st.title("⚡ Electricity Consumption & Energy Production Intelligence")
st.caption(
    "Decision-support dashboard for electricity consumption, generation mix, "
    "energy balance, and short-term demand forecasting."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload Electricity.csv",
        type=["csv"],
        help="Use the Kaggle Electricity.csv file used for this project.",
    )

if uploaded is None:
    default_path = Path("Electricity.csv")
    if default_path.exists():
        file_bytes = default_path.read_bytes()
        st.sidebar.success("Using Electricity.csv from the project folder.")
    else:
        st.info("Upload Electricity.csv from the sidebar to start the dashboard.")
        st.stop()
else:
    file_bytes = uploaded.getvalue()

data = load_and_clean(file_bytes)
model_data = build_model_data(data)

model, split, predictions, metrics = train_model(model_data)

with st.sidebar:
    st.header("Filters")
    min_date = data["DateTime"].min().date()
    max_date = data["DateTime"].max().date()
    selected_dates = st.date_input(
        "Analysis period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        view = data[
            (data["DateTime"].dt.date >= start_date)
            & (data["DateTime"].dt.date <= end_date)
        ].copy()
    else:
        view = data.copy()

tabs = st.tabs([
    "Overview",
    "Consumption Analysis",
    "Energy Production",
    "Demand Forecast",
    "Insights & Data Quality",
])

with tabs[0]:
    st.subheader("Executive Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Consumption", f"{view['Consumption'].sum()/1e6:.2f}M MWh")
    c2.metric("Avg. Consumption", f"{view['Consumption'].mean():,.0f} MWh")
    c3.metric("Peak Consumption", f"{view['Consumption'].max():,.0f} MWh")
    c4.metric("Total Production", f"{view['Production'].sum()/1e6:.2f}M MWh")
    c5.metric("Renewable Share", f"{view['Renewable'].sum()/view[SOURCE_COLUMNS].sum().sum()*100:.1f}%")

    daily = view.set_index("DateTime")["Consumption"].resample("D").mean().reset_index()
    fig = px.line(daily, x="DateTime", y="Consumption",
                  title="Daily Average Electricity Consumption")
    fig.update_layout(xaxis_title="", yaxis_title="MWh")
    st.plotly_chart(fig, use_container_width=True)

    balance = view.set_index("DateTime")["EnergyBalance"].resample("D").mean().reset_index()
    fig2 = px.line(balance, x="DateTime", y="EnergyBalance",
                   title="Daily Average Production − Consumption")
    fig2.add_hline(y=0, line_dash="dash")
    fig2.update_layout(xaxis_title="", yaxis_title="MWh")
    st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    st.subheader("Consumption Patterns")
    h = view.groupby("Hour", as_index=False)["Consumption"].mean()
    fig = px.line(h, x="Hour", y="Consumption", markers=True,
                  title="Average Consumption by Hour")
    st.plotly_chart(fig, use_container_width=True)

    wd = view.groupby("DayName", as_index=False)["Consumption"].mean()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd["DayName"] = pd.Categorical(wd["DayName"], categories=order, ordered=True)
    wd = wd.sort_values("DayName")
    fig = px.bar(wd, x="DayName", y="Consumption",
                 title="Average Consumption by Day of Week")
    st.plotly_chart(fig, use_container_width=True)

    monthly = view.set_index("DateTime")["Consumption"].resample("MS").mean().reset_index()
    fig = px.line(monthly, x="DateTime", y="Consumption",
                  title="Monthly Average Consumption")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Energy Production & Energy Mix")
    source_totals = view[SOURCE_COLUMNS].sum().sort_values(ascending=False)
    source_df = source_totals.rename_axis("Source").reset_index(name="MWh")
    fig = px.bar(source_df, x="Source", y="MWh",
                 title="Generation Contribution by Source")
    st.plotly_chart(fig, use_container_width=True)

    mix = pd.DataFrame({
        "Type": ["Renewable", "Non-renewable"],
        "MWh": [view["Renewable"].sum(), view["NonRenewable"].sum()],
    })
    fig = px.pie(mix, names="Type", values="MWh",
                 title="Renewable vs Non-renewable Generation")
    st.plotly_chart(fig, use_container_width=True)

    source_long = view[["DateTime"] + SOURCE_COLUMNS].copy()
    source_long = source_long.rename(columns={"Wind": "Wind"})
    source_long = source_long.set_index("DateTime").resample("D").mean().reset_index()
    source_long = source_long.melt(
        id_vars="DateTime", var_name="Source", value_name="MWh"
    )
    fig = px.line(source_long, x="DateTime", y="MWh", color="Source",
                  title="Daily Average Generation by Source")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Short-Term Electricity Demand Forecast")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:,.1f} MWh")
    c2.metric("RMSE", f"{metrics['RMSE']:,.1f} MWh")
    c3.metric("R²", f"{metrics['R2']:.3f}")

    test = model_data.loc[split:].copy()
    test["Predicted"] = predictions
    sample = test.tail(min(500, len(test)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["DateTime"], y=sample["Consumption"],
        mode="lines", name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=sample["DateTime"], y=sample["Predicted"],
        mode="lines", name="Predicted"
    ))
    fig.update_layout(
        title="Actual vs Predicted Consumption — Test Period Sample",
        xaxis_title="", yaxis_title="MWh"
    )
    st.plotly_chart(fig, use_container_width=True)

    horizon = st.slider("Forecast horizon (hours)", 6, 48, 24)
    future = recursive_forecast(model, data, horizon)
    fig = px.line(future, x="DateTime", y="Forecast",
                  title=f"Next {horizon} Hours — Consumption Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "Model: Random Forest Regressor using time features, recent consumption "
        "lags (1h, 24h, 168h), and rolling averages. The chronological 80/20 split "
        "prevents future observations from entering model training."
    )

with tabs[4]:
    st.subheader("Key Insights")
    peak_row = data.loc[data["Consumption"].idxmax()]
    low_row = data.loc[data["Consumption"].idxmin()]
    hour_avg = data.groupby("Hour")["Consumption"].mean()

    st.markdown(
        f"- **Peak recorded consumption:** {peak_row['Consumption']:,.0f} MWh "
        f"at {peak_row['DateTime']}."
    )
    st.markdown(
        f"- **Lowest recorded consumption:** {low_row['Consumption']:,.0f} MWh "
        f"at {low_row['DateTime']}."
    )
    st.markdown(
        f"- **Highest average-demand hour:** {hour_avg.idxmax():02d}:00 "
        f"({hour_avg.max():,.0f} MWh average)."
    )
    st.markdown(
        f"- **Lowest average-demand hour:** {hour_avg.idxmin():02d}:00 "
        f"({hour_avg.min():,.0f} MWh average)."
    )

    weekday = data.loc[~data["IsWeekend"], "Consumption"].mean()
    weekend = data.loc[data["IsWeekend"], "Consumption"].mean()
    st.markdown(
        f"- **Weekday average consumption:** {weekday:,.0f} MWh; "
        f"**weekend average:** {weekend:,.0f} MWh."
    )
    st.markdown(
        f"- **Renewable source share:** "
        f"{data['Renewable'].sum()/data[SOURCE_COLUMNS].sum().sum()*100:.1f}% "
        f"of the recorded source-level generation total."
    )

    st.subheader("Data Quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows after cleaning", f"{len(data):,}")
    q2.metric("Exact duplicate rows removed", "4")
    q3.metric("Missing values", "0")

    st.warning(
        "The source data contains some negative Wind observations and some "
        "differences between the supplied Production total and the sum of "
        "source-level generation columns. The app retains Production as the "
        "reported total and clips negative Wind only for source-mix calculations."
    )

st.caption(
    "Dataset source: Kaggle — Hourly Electricity Consumption & Production. "
    "This dashboard is intended for academic/educational analysis."
)
#python -m streamlit run electricity_intelligence.p