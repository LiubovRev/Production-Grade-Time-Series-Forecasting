# 📈 Production-Grade Time Series Forecasting Pipeline

**From Naive Baselines to Machine Learning Deployment**

This project demonstrates a **professional end-to-end forecasting workflow**. It moves beyond simple curve-fitting to explore how external signals (covariates) and machine learning (XGBoost) can significantly improve business-critical predictions.

The core of this repository is a robust, model-agnostic pipeline designed for scalability, transparency, and deployment.

---

## 🚀 Key Highlights

- ✅ **Temporal Validation**: Strict train/test splitting to prevent look-ahead bias.
- ✅ **Rigorous Benchmarking**: Complex models are only accepted if they outperform "Naive" baselines.
- ✅ **Advanced Feature Engineering**: Implementation of cyclical date encoding (Sine/Cosine) to capture seasonal nuances.
- ✅ **Optimized XGBoost**: Hyperparameter tuning via grid search focused on minimizing MAE.
- ✅ **Production Ready**: Full serialization of the best model and a **Streamlit** dashboard for end-users.

---

## 🏗️ Project Structure

```text
├── data/                        # Raw and processed datasets (CSV)
├── models/                      # Saved .pth model weights and metadata
├── images/                      # Visualization exports for documentation
├── time_series_analysis.ipynb   # Main research and training pipeline
├── utils.py                     # Core logic: evaluation, plotting, and grid search
├── app.py                       # Streamlit application code
└── README.md                    # Project documentation
```

## 🔍 Methodology
**1. Exploratory Data Analysis (EDA)**

We decompose the time series to isolate the Trend, Weekly Seasonality, and Residuals. This step informs our lag selection (e.g., using a 7-day lag to match strong weekly patterns found in the ACF/PACF analysis).

**2. Feature Engineering ⭐**

Traditional date features (like day of the month 1-31) fail to capture the "closeness" of December to January. To fix this, we implement:  
- Cyclical Encoding: Using Sine/Cosine Transforms to map dates to a circular space, ensuring the model understands temporal continuity.  
- Binary Flags: Weekend vs. Weekday detection to help the model account for repeating demand spikes.

**3. Model Selection & Tuning**

We evaluate and compare three distinct tiers of models to find the optimal balance between complexity and accuracy:  
- Baselines: Naive Seasonal (7-day window).  
- Statistical: ARIMA and Exponential Smoothing (ETS).  
- Machine Learning: XGBoost (via the Darts library), leveraging both past lags and future cyclical covariates.

## 📊 Results Summary

After conducting a constrained grid search across 30 combinations of lags, tree depth, and learning rates, the results are summarized below:
Model	                      MAE	    MAPE
Naive Seasonal (Baseline)	  6.1196  38.17%
XGBoost (Tuned)            3.9199	  ~23.5%

**Key Insight:** The tuned XGBoost model achieved a 36% reduction in error over the baseline, justifying the use of machine learning for this specific dataset.
---

