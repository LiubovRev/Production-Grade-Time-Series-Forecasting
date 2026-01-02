import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import XGBModel
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Demand Forecasting Dashboard")
st.markdown("This application uses an optimized XGBoost model with cyclical covariates to predict future sales.")

# 1. Load Data & Model
@st.cache_resource
def load_assets():
    model = XGBModel.load("models/xgb_best_model.pth")
    # Load context data for prediction
    train_df = pd.read_csv("data/last_train_data.csv", parse_dates=['date'])
    cov_df = pd.read_csv("data/full_covariates.csv", parse_dates=['date'])
    
    train_ts = TimeSeries.from_dataframe(train_df, time_col='date', value_cols='sales')
    cov_ts = TimeSeries.from_dataframe(cov_df, time_col='date', value_cols=['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend'])
    
    return model, train_ts, cov_ts

try:
    model, train_ts, cov_ts = load_assets()
    
    # 2. Sidebar configuration
    st.sidebar.header("Forecast Settings")
    horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 90, 30)

    # 3. Prediction logic
    if st.button("Generate Forecast"):
        with st.spinner('Calculating...'):
            forecast = model.predict(n=horizon, past_covariates=cov_ts)
            
            # 4. Visualization
            fig = go.Figure()
            # Show last 90 days of history for context
            hist_subset = train_ts[-90:]
            
            fig.add_trace(go.Scatter(x=hist_subset.time_index, y=hist_subset.values().flatten(), 
                                     name="History", line=dict(color="#2E86C1")))
            fig.add_trace(go.Scatter(x=forecast.time_index, y=forecast.values().flatten(), 
                                     name="Forecast", line=dict(color="#E67E22", dash='dash')))
            
            fig.update_layout(title=f"Sales Prediction for the next {horizon} days",
                              xaxis_title="Date", yaxis_title="Units Sold",
                              template="plotly_white", hovermode="x unified")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Metrics/Summary
            col1, col2 = st.columns(2)
            col1.metric("Total Expected Sales", int(forecast.values().sum()))
            col2.metric("Peak Demand", int(forecast.values().max()))

except FileNotFoundError:
    st.error("Model file not found. Please run the notebook first to save the model.")