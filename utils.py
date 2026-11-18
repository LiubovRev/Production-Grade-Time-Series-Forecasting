"""Shared helpers for the forecasting notebook and the Streamlit app."""

import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.metrics import mae, mape
from darts.models import XGBModel

COVARIATE_COLS = ["dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]


def load_series(csv_path, store: int = 1, item: int = 1) -> TimeSeries:
    """Daily sales for one store/item pair as a Darts TimeSeries."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    subset = df[(df["store"] == store) & (df["item"] == item)]
    return TimeSeries.from_dataframe(
        subset, time_col="date", value_cols="sales", freq="D"
    ).astype(np.float32)


def make_calendar_covariates(start, end) -> TimeSeries:
    """Cyclical calendar features for every day in [start, end].

    Calendar features are known in advance, so they are used as
    *future* covariates and can be generated for any forecast horizon.
    """
    idx = pd.date_range(start, end, freq="D")
    dow, month = idx.dayofweek, idx.month
    df = pd.DataFrame(
        {
            "date": idx,
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "is_weekend": (dow >= 5).astype(float),
        }
    )
    return TimeSeries.from_dataframe(df, time_col="date", value_cols=COVARIATE_COLS).astype(np.float32)


def make_xgb(params: dict, random_state: int = 42) -> XGBModel:
    """XGBoost forecaster on target lags plus same-day calendar covariates."""
    return XGBModel(
        lags=params["lags"],
        lags_future_covariates=[0],
        output_chunk_length=params.get("output_chunk_length", 7),
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=random_state,
    )


def fit_eval_model(model, model_name: str, train: TimeSeries, target: TimeSeries, future_covariates=None):
    """Fit on `train`, forecast len(target) steps, return (MAE, MAPE %, forecast)."""
    kwargs = {"future_covariates": future_covariates} if future_covariates is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train, **kwargs)
        forecast = model.predict(len(target), **kwargs)
    mae_val, mape_val = float(mae(target, forecast)), float(mape(target, forecast))
    print(f"{model_name:<22} MAE: {mae_val:6.3f}   MAPE: {mape_val:6.2f}%")
    return mae_val, mape_val, forecast


def grid_search_xgb(train: TimeSeries, val: TimeSeries, covariates: TimeSeries, param_grid: dict):
    """Exhaustive search on the validation window only (never on the test window)."""
    names = list(param_grid)
    rows = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(names, combo))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make_xgb(params)
            model.fit(train, future_covariates=covariates)
            forecast = model.predict(len(val), future_covariates=covariates)
        rows.append({**params, "val_MAE": float(mae(val, forecast))})
    results = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    best_params = {k: results.loc[0, k] for k in names}
    best_params = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in best_params.items()}
    print(f"Tested {len(results)} combinations. Best validation MAE: {results.loc[0, 'val_MAE']:.3f} with {best_params}")
    return best_params, results


def plot_forecasts(history: TimeSeries, actual: TimeSeries = None, preds: dict = None, title: str = "Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.time_index, y=history.values().flatten(), mode="lines", name="History"))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.time_index, y=actual.values().flatten(), mode="lines", name="Actual"))
    for name, pred in (preds or {}).items():
        fig.add_trace(go.Scatter(x=pred.time_index, y=pred.values().flatten(), mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Units sold",
                      hovermode="x unified", template="plotly_white")
    return fig
