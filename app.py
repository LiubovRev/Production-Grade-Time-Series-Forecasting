from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from darts import TimeSeries

from utils import make_calendar_covariates, plot_forecasts

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "xgb_final_model.joblib"
HISTORY_PATH = BASE_DIR / "data" / "history.csv"

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Demand Forecasting Dashboard")


@st.cache_resource
def load_assets():
    bundle = joblib.load(MODEL_PATH)
    history_df = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    history = TimeSeries.from_dataframe(history_df, time_col="date", value_cols="sales", freq="D").astype("float32")
    return bundle, history


if not MODEL_PATH.exists() or not HISTORY_PATH.exists():
    st.error("Model or history not found. Run time_series_analysis.ipynb first.")
    st.stop()

bundle, history = load_assets()
model = bundle["model"]

st.markdown(
    f"XGBoost with lagged sales and cyclical calendar covariates, trained on the full history "
    f"({history.start_time().date()} to {history.end_time().date()}) for store {bundle['store']}, item {bundle['item']}. "
    "Forecasts start the day after the last observation."
)

horizon = st.sidebar.slider("Forecast horizon (days)", 1, 90, 30)

if st.button("Generate forecast"):
    with st.spinner("Forecasting..."):
        covariates = make_calendar_covariates(history.start_time(), pd.Timestamp(history.end_time()) + pd.Timedelta(days=horizon))
        forecast = model.predict(n=horizon, series=history, future_covariates=covariates)

    fig = plot_forecasts(history[-90:], preds={"Forecast": forecast},
                         title=f"Sales forecast for the next {horizon} days")
    st.plotly_chart(fig, width="stretch")

    values = forecast.values().flatten()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total expected sales", f"{values.sum():.0f}")
    col2.metric("Average per day", f"{values.mean():.1f}")
    col3.metric("Peak day", f"{values.max():.0f}")

    with st.expander("Raw forecast"):
        st.dataframe(
            pd.DataFrame({"date": forecast.time_index, "predicted_sales": values.round(1)}),
            width="stretch",
        )
