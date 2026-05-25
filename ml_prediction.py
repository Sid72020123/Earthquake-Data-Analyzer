"""
Machine Learning module for predicting earthquake frequency trends.

This module uses a Moving Average baseline to forecast earthquake frequency
trends over time. The Moving Average is simple, robust to noise, and
interpretable for monthly earthquake counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
from sklearn.metrics import mean_squared_error, r2_score


def get_ml_data_from_full_history(data, years=5):
    """
    Extract ML training data from full earthquake history (last N years).

    Uses the full historical dataset instead of filtered dashboard data,
    ensuring consistent ML training regardless of dashboard filters.

    Parameters:
    -----------
    data : pd.DataFrame
        Full earthquake data with 'time' column
    years : int
        Number of years of historical data to use (default: 5)

    Returns:
    --------
    pd.DataFrame
        Filtered data from last N years
    """

    df = data.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Get the most recent date in the dataset
    max_date = df["time"].max()

    # Calculate cutoff date (N years ago)
    cutoff_date = max_date - pd.DateOffset(years=years)

    # Filter to last N years
    df_filtered = df[df["time"] >= cutoff_date].copy()

    return df_filtered


def prepare_time_series_data(data, period="M"):
    """
    Prepare earthquake frequency data grouped by time period.

    Monthly aggregation provides stable trends while reducing noise from
    short-term fluctuations. This is better for trend forecasting than
    weekly or daily data which can be too noisy.

    Parameters:
    -----------
    data : pd.DataFrame
        Earthquake data with 'time' column (must be datetime)
    period : str
        Pandas period string ('D' for daily, 'W' for weekly, 'M' for monthly)
        Default is 'M' (monthly) for smoother trends

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'time' and 'count' columns
    """
    # Create a copy to avoid modifying original data
    df = data.copy()

    # Ensure time is datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Convert to timezone-naive to avoid warnings with to_period()
    df["time"] = df["time"].dt.tz_localize(None)

    # Group by period and count earthquakes
    frequency_data = (
        df.groupby(df["time"].dt.to_period(period)).size().reset_index(name="count")
    )

    # Convert period to timestamp (start of period)
    frequency_data["time"] = frequency_data["time"].dt.to_timestamp()
    frequency_data = frequency_data.sort_values("time").reset_index(drop=True)

    return frequency_data


def convert_dates_to_numeric(dates):
    """
    Convert datetime values to numeric values for ML model.

    This converts dates to 'days since the earliest date' so the model
    can learn the relationship between time and earthquake frequency.

    Parameters:
    -----------
    dates : pd.Series or np.ndarray
        Datetime values to convert

    Returns:
    --------
    np.ndarray
        Numeric representation of dates (days since minimum date)
    float
        The minimum date timestamp (for reference)
    """
    # Convert to datetime if not already
    dates = pd.to_datetime(dates)

    # Get the minimum date as reference
    min_date = dates.min()

    # Convert to days since minimum date
    numeric_dates = (dates - min_date).dt.total_seconds() / (24 * 3600)

    return numeric_dates.values, min_date.timestamp()


def train_ml_model(frequency_data, test_size=0.2, window=3, random_state=42):
    """
    Train a Hybrid Model (Exponential Smoothing + Random Forest) to predict earthquake frequency.

    The Hybrid Model is robust for noisy time series like earthquake monthly counts.
    It combines a base trend model with a machine learning model to predict residuals.

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns
    test_size : float
        Fraction of data to use for testing (default 0.2)
    window : int
        Window size (months) for the moving average (default 3)

    Returns:
    --------
    dict
        Dictionary containing model, data, and performance metrics
    """
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Prepare arrays
    dates = pd.to_datetime(frequency_data["time"])
    y = frequency_data["count"].values

    # Chronological split
    split_index = max(1, int(len(y) * (1 - test_size)))
    y_train = y[:split_index]
    y_test = y[split_index:]
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    # Safety check
    if len(y_test) == 0:
        y_train = y[:-1]
        y_test = y[-1:]
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    # Base Model: Exponential Smoothing
    try:
        base_model = ExponentialSmoothing(
            y_train, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        base_train_pred = base_model.fittedvalues
        base_test_pred = base_model.forecast(steps=len(y_test))
    except Exception:
        # Fallback to mean if ES fails
        base_model = None
        base_train_pred = np.full(len(y_train), np.mean(y_train))
        base_test_pred = np.full(len(y_test), np.mean(y_train))

    # Calculate residuals
    residuals_train = y_train - base_train_pred

    # Features for ML Model (Random Forest)
    # Using time index and month to capture any non-linear trend or seasonality
    X_train_ml = pd.DataFrame(
        {"time_idx": X_train.flatten(), "month": dates.iloc[:split_index].dt.month}
    )

    X_test_ml = pd.DataFrame(
        {"time_idx": X_test.flatten(), "month": dates.iloc[split_index:].dt.month}
    )

    # ML Model: Random Forest to predict residuals
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=random_state
    )
    rf_model.fit(X_train_ml, residuals_train)

    rf_train_pred = rf_model.predict(X_train_ml)
    rf_test_pred = rf_model.predict(X_test_ml)

    # Final Predictions (Base + ML Residuals)
    y_train_pred = base_train_pred + rf_train_pred
    y_test_pred = base_test_pred + rf_test_pred

    # Ensure no negative predictions (counts can't be negative)
    y_train_pred = np.maximum(y_train_pred, 0)
    y_test_pred = np.maximum(y_test_pred, 0)

    # Simple model object with forecast method for compatibility
    class HybridModel:
        def __init__(self, base, rf, last_train_idx, last_date):
            self.base = base
            self.rf = rf
            self.last_train_idx = last_train_idx
            self.last_date = pd.to_datetime(last_date)

        def forecast(self, steps=1):
            try:
                base_f = self.base.forecast(steps=steps)
            except Exception:
                base_f = np.full(steps, np.mean(y_train))

            future_dates = pd.date_range(
                start=self.last_date + pd.DateOffset(months=1), periods=steps, freq="MS"
            )

            X_f_ml = pd.DataFrame(
                {
                    "time_idx": np.arange(
                        self.last_train_idx + 1, self.last_train_idx + 1 + steps
                    ),
                    "month": future_dates.month,
                }
            )

            rf_f = self.rf.predict(X_f_ml)

            return np.maximum(base_f + rf_f, 0)

    model = HybridModel(
        base_model, rf_model, len(y_train) + len(y_test) - 1, dates.iloc[-1]
    )

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Feature importance (time index importance from RF)
    feat_imp = float(rf_model.feature_importances_[0])

    # Dummy poly/scaler kept for API compatibility
    poly = type("obj", (object,), {"transform": lambda x: x})()
    scaler = type("obj", (object,), {"transform": lambda x: x})()

    results = {
        "model": model,
        "poly": poly,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "frequency_data": frequency_data,
        "feature_importance": feat_imp,
        "min_date_timestamp": dates.min().timestamp(),
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_total_samples": len(y_train) + len(y_test),
    }

    return results


def plot_actual_vs_predicted(results):
    """
    Create a matplotlib figure showing actual vs predicted earthquake frequency.

    Shows two views:
    1. Time series comparison: How well the model follows the actual trend
    2. Scatter plot: Prediction accuracy (points near diagonal = good predictions)

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with actual vs predicted comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Combine all data for visualization
    X_all = np.vstack([results["X_train"], results["X_test"]])
    y_all = np.hstack([results["y_train"], results["y_test"]])
    y_pred_all = np.hstack([results["y_train_pred"], results["y_test_pred"]])

    # Sort by time for better visualization
    sorted_idx = np.argsort(X_all.flatten())
    X_sorted = X_all[sorted_idx].flatten()
    y_sorted = y_all[sorted_idx]
    y_pred_sorted = y_pred_all[sorted_idx]

    # Plot 1: Time series comparison
    ax1.plot(
        X_sorted,
        y_sorted,
        "o-",
        label="Actual Frequency",
        alpha=0.7,
        linewidth=2,
        markersize=6,
        color="#0ea5a4",
    )
    ax1.plot(
        X_sorted,
        y_pred_sorted,
        "s--",
        label="Hybrid Model Trend",
        alpha=0.7,
        linewidth=2,
        markersize=6,
        color="#f97316",
    )
    ax1.set_xlabel("Time (days since start)", fontsize=11)
    ax1.set_ylabel("Earthquake Frequency (count)", fontsize=11)
    ax1.set_title("Trend Analysis: Actual vs Predicted", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot for prediction accuracy
    ax2.scatter(y_all, y_pred_all, alpha=0.6, s=50, color="#0ea5a4")

    # Add perfect prediction line (for reference)
    min_val = min(y_all.min(), y_pred_all.min())
    max_val = max(y_all.max(), y_pred_all.max())
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Perfect Prediction",
        linewidth=2,
    )

    ax2.set_xlabel("Actual Frequency", fontsize=11)
    ax2.set_ylabel("Predicted Frequency", fontsize=11)
    ax2.set_title("Prediction Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_prediction_plotly(results, future_periods=12):
    """
    Create a Plotly interactive figure with predictions and future forecast.

    Shows:
    - Raw earthquake data (noisy)
    - Model predictions (smooth trend)
    - Future forecast (what the model predicts coming)

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()
    future_periods : int
        Number of future months to forecast (default: 12)

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    frequency_data = results["frequency_data"].copy()
    model = results["model"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_train_pred = results["y_train_pred"]
    y_test_pred = results["y_test_pred"]

    # Prepare historical data
    frequency_data["time"] = pd.to_datetime(frequency_data["time"])
    all_dates = pd.to_datetime(frequency_data["time"])

    # Combine all predictions
    historical_predictions = np.concatenate([y_train_pred, y_test_pred])
    actual_counts = np.concatenate([y_train, y_test])

    # Generate future predictions using the Moving Average model
    last_date = all_dates.iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_periods,
        freq="MS",
    )

    # Forecast future values (moving average model provides forecast method)
    future_y_pred = model.forecast(steps=future_periods)

    # Create interactive figure
    fig = go.Figure()

    # Add raw data (lightly - to show it's noisy)
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=actual_counts,
            mode="markers",
            name="Raw Data (Noisy)",
            marker=dict(size=6, color="#94a3b8"),
            opacity=0.5,
        )
    )

    # Add model predictions for historical period
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=historical_predictions,
            mode="lines",
            name="Hybrid Model Trend",
            line=dict(color="#0ea5a4", width=3),
        )
    )

    # Add future forecast
    fig.add_trace(
        go.Scatter(
            x=list(all_dates) + list(future_dates),
            y=list(historical_predictions) + list(future_y_pred),
            mode="lines",
            name="Future Forecast",
            line=dict(color="#ec4899", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Earthquake Frequency Trend Analysis with Future Forecast",
        xaxis_title="Date",
        yaxis_title="Earthquake Frequency (monthly count)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        font=dict(size=11),
    )

    return fig


def create_feature_importance_plot(results):
    """
    Create a simple visualization of feature importance.

    A simple plot showing the importance of the time feature (only feature used).

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with feature importance
    """
    # More useful feature-summary visualization for a single-feature time series.
    # Left: scatter of time vs counts with simple linear fit and Pearson r
    # Right: average counts by calendar month (seasonality check)
    freq = results.get("frequency_data")
    if freq is None:
        # Fallback to simple bar if data missing
        fig, ax = plt.subplots(figsize=(8, 3))
        importance = results.get("feature_importance", 1.0)
        ax.barh(["Time"], [importance], color="#0ea5a4")
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Feature Importance (Time)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.1)
        ax.text(importance + 0.02, 0, f"{importance:.3f}", va="center", fontsize=11)
        plt.tight_layout()
        return fig

    df = freq.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Numeric time for regression (days since start)
    t0 = df["time"].min()
    numeric_time = (df["time"] - t0).dt.total_seconds() / (24 * 3600)
    counts = df["count"].values

    # Linear fit (simple trend) for visualization
    try:
        coef = np.polyfit(numeric_time, counts, 1)
        fit_line = np.polyval(coef, numeric_time)
        # Pearson-like measure (correlation)
        if np.std(counts) > 0 and np.std(numeric_time) > 0:
            corr = np.corrcoef(numeric_time, counts)[0, 1]
        else:
            corr = 0.0
    except Exception:
        fit_line = np.full_like(counts, np.mean(counts))
        corr = 0.0

    # Monthly seasonality: average by calendar month (1..12)
    df["month"] = df["time"].dt.month
    month_avg = df.groupby("month")["count"].mean().reindex(range(1, 13), fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter + trend
    ax1.scatter(df["time"], counts, alpha=0.6, s=30, color="#0ea5a4")
    ax1.plot(df["time"], fit_line, color="#f97316", linewidth=2, label="Linear trend")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Earthquake Count", fontsize=10)
    ax1.set_title("Time vs Count (trend + correlation)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    ax1.text(
        0.02,
        0.95,
        f"corr(time,count) = {corr:.3f}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    # Month averages
    ax2.bar(month_avg.index, month_avg.values, color="#60a5fa")
    ax2.set_xlabel("Month", fontsize=10)
    ax2.set_ylabel("Avg Earthquake Count", fontsize=10)
    ax2.set_title("Average Count by Calendar Month")
    ax2.set_xticks(range(1, 13))
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    return fig


def get_model_explanation():
    """
    Return a beginner-friendly explanation of the Hybrid model.

    Returns:
    --------
    str
        Explanation text in markdown format
    """
    explanation = """
    ### 🤖 About This Hybrid ML Model
    
    **What it predicts:**
    Earthquake **frequency trends** over time using a Hybrid approach.
    
    **What it does NOT predict:**
    - Exact earthquake locations or times
    - Specific earthquake magnitudes  
    - When earthquakes will occur
    
    **Key Features:**
    - **Data**: Last 5 years of global earthquake data (~60 months)
    - **Method**: Hybrid Model (Exponential Smoothing + Random Forest)
    - **Aggregation**: Monthly earthquake counts for stability
    - **Train/Test Split**: 80% training (~48 months), 20% testing (~12 months)
    
    **Why a Hybrid Model?**
    - Earthquake frequency is **highly chaotic and noisy**
    - **Exponential Smoothing** captures the underlying trend and baseline levels
    - **Random Forest** learns from the residuals (errors) of the base model to find complex non-linear patterns (like seasonality or hidden correlations)
    - Combining them yields higher accuracy than a single simple model
    
    **How it works (simplified):**
    1. Base model calculates the general trend of earthquakes.
    2. We measure the "errors" (residuals) between actual data and base trend.
    3. Random Forest tries to predict these errors using time features.
    4. Final prediction = Base Trend + Random Forest Adjustment.
    
    **What the model shows:**
    - **Teal line**: The predicted Hybrid trend
    - **Pink dotted line**: The n-month future forecast
    - **Gray dots**: Raw monthly data (noisy)
    
    **Important Limitations:**
    - Earthquake patterns are inherently unpredictable
    - Model captures statistical patterns only, not physical drivers
    
    **Understanding R² Score:**
    - Ranges from 0 to 1 (higher = better fit)
    - Negative: Model worse than simple average (indicative of randomness)
    """
    return explanation


def compare_models(frequency_data):
    """
    Compare multiple ML models on the same earthquake frequency data.

    Trains and evaluates 7 different models: Linear Regression, Exponential
    Smoothing, Moving Average, Ridge Regression, ARIMA, Seasonal Naive, and
    Baseline (mean prediction). Returns comparison metrics sorted by Test R².

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns

    Returns:
    --------
    pd.DataFrame
        Comparison table with columns: Model, Train R², Test R², Train RMSE, Test RMSE
        Sorted by Test R² (highest first)
    """
    try:
        from sklearn.linear_model import Ridge
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        # Return empty dataframe if statsmodels not available
        return pd.DataFrame()

    # Prepare data
    y = frequency_data["count"].values
    split_index = int(len(y) * 0.8)
    y_train = y[:split_index]
    y_test = y[split_index:]

    results_list = []

    # 1. Hybrid Model (Current model from train_ml_model)
    try:
        ml_results = train_ml_model(frequency_data, test_size=0.2)
        results_list.append(
            {
                "Model": "Hybrid (Exp. Smoothing + RF)",
                "Train R²": ml_results["train_r2"],
                "Test R²": ml_results["test_r2"],
                "Train RMSE": ml_results["train_rmse"],
                "Test RMSE": ml_results["test_rmse"],
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Hybrid (Exp. Smoothing + RF)",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 2. Exponential Smoothing
    try:
        exp_smooth = ExponentialSmoothing(
            y_train, trend="add", seasonal=None, initialization_method="estimated"
        )
        exp_model = exp_smooth.fit(optimized=True)
        exp_train_pred = exp_model.fittedvalues
        exp_test_pred = exp_model.forecast(steps=len(y_test))
        exp_train_r2 = r2_score(y_train, exp_train_pred)
        exp_test_r2 = r2_score(y_test, exp_test_pred)
        exp_train_rmse = np.sqrt(mean_squared_error(y_train, exp_train_pred))
        exp_test_rmse = np.sqrt(mean_squared_error(y_test, exp_test_pred))
        results_list.append(
            {
                "Model": "Exponential Smoothing",
                "Train R²": exp_train_r2,
                "Test R²": exp_test_r2,
                "Train RMSE": exp_train_rmse,
                "Test RMSE": exp_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Exponential Smoothing",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 3. Moving Average (simple implementation for comparison)
    try:
        window_size = 3
        ma_train = np.convolve(
            y_train, np.ones(window_size) / window_size, mode="valid"
        )
        ma_train_actual = y_train[window_size - 1 :]
        ma_test_pred = np.full(len(y_test), np.mean(y_train))
        ma_train_r2 = r2_score(ma_train_actual, ma_train)
        ma_test_r2 = r2_score(y_test, ma_test_pred)
        ma_train_rmse = np.sqrt(mean_squared_error(ma_train_actual, ma_train))
        ma_test_rmse = np.sqrt(mean_squared_error(y_test, ma_test_pred))
        results_list.append(
            {
                "Model": "Moving Average",
                "Train R²": ma_train_r2,
                "Test R²": ma_test_r2,
                "Train RMSE": ma_train_rmse,
                "Test RMSE": ma_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Moving Average",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 4. Ridge Regression
    try:
        X_train_idx = np.arange(len(y_train)).reshape(-1, 1)
        X_test_idx = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_idx, y_train)
        ridge_train_pred = ridge_model.predict(X_train_idx)
        ridge_test_pred = ridge_model.predict(X_test_idx)
        ridge_train_r2 = r2_score(y_train, ridge_train_pred)
        ridge_test_r2 = r2_score(y_test, ridge_test_pred)
        ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
        ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
        results_list.append(
            {
                "Model": "Ridge Regression",
                "Train R²": ridge_train_r2,
                "Test R²": ridge_test_r2,
                "Train RMSE": ridge_train_rmse,
                "Test RMSE": ridge_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Ridge Regression",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 5. ARIMA
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Non-invertible starting MA parameters found.*",
                category=UserWarning,
            )
            arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
        arima_train_pred = arima_model.fittedvalues
        arima_test_pred = arima_model.get_forecast(steps=len(y_test)).predicted_mean
        arima_train_r2 = r2_score(y_train, arima_train_pred)
        arima_test_r2 = r2_score(y_test, arima_test_pred)
        arima_train_rmse = np.sqrt(mean_squared_error(y_train, arima_train_pred))
        arima_test_rmse = np.sqrt(mean_squared_error(y_test, arima_test_pred))
        results_list.append(
            {
                "Model": "ARIMA",
                "Train R²": arima_train_r2,
                "Test R²": arima_test_r2,
                "Train RMSE": arima_train_rmse,
                "Test RMSE": arima_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "ARIMA",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 6. Seasonal Naive
    try:
        season = 12
        if len(y_train) >= season:
            sn_train_pred = y_train[season:]
            sn_train_actual = y_train[season:]
            sn_test_pred = np.full(
                len(y_test),
                y_train[-season] if len(y_train) >= season else np.mean(y_train),
            )
            sn_train_r2 = r2_score(sn_train_actual, sn_train_pred)
            sn_test_r2 = r2_score(y_test, sn_test_pred)
            sn_train_rmse = np.sqrt(mean_squared_error(sn_train_actual, sn_train_pred))
            sn_test_rmse = np.sqrt(mean_squared_error(y_test, sn_test_pred))
        else:
            sn_train_r2 = sn_test_r2 = sn_train_rmse = sn_test_rmse = 0.0
        results_list.append(
            {
                "Model": "Seasonal Naive",
                "Train R²": sn_train_r2,
                "Test R²": sn_test_r2,
                "Train RMSE": sn_train_rmse,
                "Test RMSE": sn_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Seasonal Naive",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # 7. Baseline (Mean)
    try:
        baseline_pred_train = np.full(len(y_train), np.mean(y_train))
        baseline_pred_test = np.full(len(y_test), np.mean(y_train))
        baseline_train_r2 = r2_score(y_train, baseline_pred_train)
        baseline_test_r2 = r2_score(y_test, baseline_pred_test)
        baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
        baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred_test))
        results_list.append(
            {
                "Model": "Baseline (Mean)",
                "Train R²": baseline_train_r2,
                "Test R²": baseline_test_r2,
                "Train RMSE": baseline_train_rmse,
                "Test RMSE": baseline_test_rmse,
            }
        )
    except Exception:
        results_list.append(
            {
                "Model": "Baseline (Mean)",
                "Train R²": 0.0,
                "Test R²": 0.0,
                "Train RMSE": 0.0,
                "Test RMSE": 0.0,
            }
        )

    # Create dataframe and sort by Test R² (highest first)
    df = pd.DataFrame(results_list)
    df = df.sort_values("Test R²", ascending=False).reset_index(drop=True)
    return df
