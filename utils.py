import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.models import NaiveSeasonal, NaiveDrift, XGBModel, ExponentialSmoothing, ARIMA, AutoARIMA, RNNModel 
from darts.metrics import mae as mae_metric, mape as mape_metric, rmse
import plotly.graph_objects as go
from functools import reduce
import itertools 
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error 

# Global warning suppression
warnings.filterwarnings('ignore')

def plot_forecasts(train: TimeSeries, val: TimeSeries = None, preds: dict = None, title: str = "Forecast Plot"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.time_index, y=train.values().flatten(), mode='lines', name='Train', line=dict(width=2)))
    if val is not None:
        fig.add_trace(go.Scatter(x=val.time_index, y=val.values().flatten(), mode='lines', name='Validation', line=dict(width=2)))
    if preds is not None:
        for name, pred in preds.items():
            fig.add_trace(go.Scatter(x=pred.time_index, y=pred.values().flatten(), mode='lines', name=name))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Sales', hovermode='x unified', template='plotly_white')
    fig.show()

def fit_eval_model(model, model_name: str, train: TimeSeries, val: TimeSeries):
    # Silencing warnings during fit/predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, 'show_warnings'): model.show_warnings = False
        model.fit(train)
        forecast = model.predict(len(val))
    
    mae_val = mae_metric(val, forecast)
    mape_val = mape_metric(val, forecast)
    print(f"{model_name} -> MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")
    return mae_val, mape_val, forecast

def append_results(results_df: pd.DataFrame, model_name: str, mae_val: float, mape_val: float):
    """
    Appends the evaluation metrics of a model to a results DataFrame.
    """
    results_df.loc[model_name] = [mae_val, mape_val]
    return results_df

def mae(actual: TimeSeries, predicted: TimeSeries):
    return mae_metric(actual, predicted)

def mape(actual: TimeSeries, predicted: TimeSeries):
    return mape_metric(actual, predicted)

def evaluate_xgb_params(train_data: TimeSeries, val_data: TimeSeries, params: dict, covariates_train: TimeSeries, covariates_full_for_pred: TimeSeries):
    # Set output_chunk_length to match validation length to avoid auto-regression warnings
    model = XGBModel(
        lags=params.get('lags', 7),
        lags_past_covariates=params.get('lags', 7),
        output_chunk_length=len(val_data), 
        n_estimators=params.get('n_estimators', 200),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        show_warnings=False, 
        random_state=42
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(series=train_data, past_covariates=covariates_train)
        forecast = model.predict(n=len(val_data), past_covariates=covariates_full_for_pred)
    
    return mae_metric(val_data, forecast), model

def simple_grid_search(train_data, val_data, covariates_train, covariates_full_for_pred, max_trials=30):
    print("Starting Simple Grid Search...")
    param_options = {
        'lags': [7, 14, 21],
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.15, 0.2]
    }
    
    best_score, best_params, best_model = float('inf'), None, None
    param_names = list(param_options.keys())
    all_combinations = list(itertools.product(*param_options.values()))[:max_trials]
    
    print(f"Testing {len(all_combinations)} parameter combinations...")
    
    for i, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        score, model = evaluate_xgb_params(train_data, val_data, params, covariates_train, covariates_full_for_pred)
        
        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
            print(f"New best score: {best_score:.4f} with params: {best_params}")
    
    print("Grid Search Complete!")
    return best_params, best_score, best_model