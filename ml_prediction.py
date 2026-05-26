"""
Machine Learning module for predicting earthquake frequency trends.

This module uses a Hybrid Model (Exponential Smoothing + Random Forest) to
forecast earthquake frequency trends. This approach combines a classical time
series model for trend with a machine learning model to capture non-linear
patterns and residuals, making it robust for noisy data like earthquake counts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
)


def _categorize_activity(arr, p33, p67):
    """Helper function to categorize continuous values into activity levels."""
    return np.array(
        ["Low" if val <= p33 else "Medium" if val <= p67 else "High" for val in arr]
    )


def _get_empty_metrics(model_name):
    """Helper function to return zeroed metrics if a model fails to train."""
    return {
        "Model": model_name,
        "Train R²": 0.0,
        "Test R²": 0.0,
        "Train RMSE": 0.0,
        "Test RMSE": 0.0,
        "Train MAE": 0.0,
        "Test MAE": 0.0,
    }


def _normalize_period(period):
    """Validate and normalize a pandas period code."""
    period = str(period).upper()
    if period not in {"D", "W", "M"}:
        raise ValueError("period must be one of 'D', 'W', or 'M'")
    return period


def _build_time_features(dates, period, start_idx=0):
    """Build time-based features for the residual model."""
    period = _normalize_period(period)
    date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)

    features = pd.DataFrame(
        {
            "time_idx": np.arange(start_idx, start_idx + len(date_series)),
            "month": date_series.dt.month,
            "quarter": date_series.dt.quarter,
        }
    )

    if period == "W":
        features["week_of_year"] = date_series.dt.isocalendar().week.astype(int)
    elif period == "D":
        features["day_of_year"] = date_series.dt.dayofyear

    return features


def _get_future_dates(last_date, steps, period):
    """Generate future timestamps that match the selected aggregation period."""
    period = _normalize_period(period)
    last_date = pd.to_datetime(last_date)

    if period == "M":
        start = last_date + pd.offsets.MonthBegin(1)
        return pd.date_range(start=start, periods=steps, freq="MS")

    if period == "W":
        start = last_date + pd.Timedelta(weeks=1)
        return pd.date_range(start=start, periods=steps, freq="7D")

    start = last_date + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=steps, freq="D")


def _period_label(period):
    """Return a human-readable label for a period code."""
    period = _normalize_period(period)
    return {"D": "daily", "W": "weekly", "M": "monthly"}[period]


def get_ml_data_from_full_history(data, years=5):
    """
    Extract ML training data from full earthquake history (last N years).

    Uses the full historical dataset instead of filtered dashboard data,
    ensuring consistent ML training regardless of dashboard filters.

    Parameters:
    -----------
    data : pd.DataFrame
        Full earthquake data with 'time' column
    years : int or None
        Number of years of historical data to use. Use None for all history.

    Returns:
    --------
    pd.DataFrame
        Filtered data from last N years
    """

    df = data.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Get the most recent date in the dataset
    max_date = df["time"].max()

    if years is None:
        df_filtered = df.copy()
    else:
        # Calculate cutoff date (N years ago)
        cutoff_date = max_date - pd.DateOffset(years=years)

        # Filter to last N years
        df_filtered = df[df["time"] >= cutoff_date].copy()

    return df_filtered.sort_values("time").reset_index(drop=True)


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
    period = _normalize_period(period)

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
    frequency_data.attrs["period"] = period

    return frequency_data


def train_ml_model(
    frequency_data, test_size=0.2, window=3, random_state=42, period="M"
):
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
    period : str
        Aggregation period used to build the input series ('D', 'W', or 'M')

    Returns:
    --------
    dict
        Dictionary containing model, data, and performance metrics
    """
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    period = _normalize_period(frequency_data.attrs.get("period", period))

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
    # Time index plus seasonality features help the residual model learn patterns.
    X_train_ml = _build_time_features(dates.iloc[:split_index], period, start_idx=0)
    X_test_ml = _build_time_features(
        dates.iloc[split_index:], period, start_idx=len(y_train)
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
        def __init__(self, base, rf, last_train_idx, last_date, period):
            self.base = base
            self.rf = rf
            self.last_train_idx = last_train_idx
            self.last_date = pd.to_datetime(last_date)
            self.period = period

        def forecast(self, steps=1):
            try:
                base_f = self.base.forecast(steps=steps)
            except Exception:
                base_f = np.full(steps, np.mean(y_train))

            future_dates = _get_future_dates(self.last_date, steps, self.period)

            X_f_ml = _build_time_features(
                future_dates,
                self.period,
                start_idx=self.last_train_idx + 1,
            )

            rf_f = self.rf.predict(X_f_ml)

            return np.maximum(base_f + rf_f, 0)

    model = HybridModel(
        base_model,
        rf_model,
        len(y_train) + len(y_test) - 1,
        dates.iloc[-1],
        period,
    )

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Additional error metrics
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 1))) * 100
    test_max_error = np.max(np.abs(y_test - y_test_pred))

    results = {
        "model": model,
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
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "test_max_error": test_max_error,
        "frequency_data": frequency_data,
        "period": period,
        "min_date_timestamp": dates.min().timestamp(),
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_total_samples": len(y_train) + len(y_test),
    }

    return results


def plot_actual_vs_predicted(results):
    """
    Create a Plotly figure showing actual vs predicted earthquake frequency.

    Shows two views:
    1. Time series comparison: How well the model follows the actual trend
    2. Scatter plot: Prediction accuracy (points near diagonal = good predictions)

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with actual vs predicted comparison
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Trend Analysis: Actual vs Predicted", "Prediction Accuracy"),
    )

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
    fig.add_trace(
        go.Scatter(
            x=X_sorted,
            y=y_sorted,
            mode="lines+markers",
            name="Actual Frequency",
            marker=dict(color="#0ea5a4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=X_sorted,
            y=y_pred_sorted,
            mode="lines+markers",
            name="Hybrid Model Trend",
            line=dict(dash="dash", color="#f97316"),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Time (days since start)", row=1, col=1)
    fig.update_yaxes(title_text="Earthquake Frequency (count)", row=1, col=1)

    # Plot 2: Scatter plot for prediction accuracy
    fig.add_trace(
        go.Scatter(
            x=y_all,
            y=y_pred_all,
            mode="markers",
            name="Predictions",
            marker=dict(color="#0ea5a4", opacity=0.6, size=8),
        ),
        row=1,
        col=2,
    )

    # Add perfect prediction line (for reference)
    min_val = min(y_all.min(), y_pred_all.min())
    max_val = max(y_all.max(), y_pred_all.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash", color="red"),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Actual Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Frequency", row=1, col=2)

    fig.update_layout(
        height=450,
        showlegend=True,
        margin=dict(t=50, b=20, l=20, r=20),
        template="plotly_white",
    )
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
    period = results.get("period", frequency_data.attrs.get("period", "M"))
    period_label = _period_label(period)
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

    # Generate future predictions using the selected aggregation step
    last_date = all_dates.iloc[-1]
    future_dates = _get_future_dates(last_date, future_periods, period)

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
        yaxis_title=f"Earthquake Frequency ({period_label} count)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        font=dict(size=11),
    )

    return fig


def plot_residuals(results):
    """
    Create a Residuals Analysis plot to act as a regression alternative
    to a classification Confusion Matrix. Shows where the model makes errors.
    """
    y_test = results["y_test"]
    y_pred = results["y_test_pred"]
    residuals = y_test - y_pred

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Error Distribution", "Residuals vs Predicted")
    )

    # 1. Residual Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals, name="Residuals", marker_color="#ef4444", opacity=0.7
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.update_xaxes(title_text="Error (Actual - Predicted)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)

    # 2. Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals vs Pred",
            marker=dict(color="#3b82f6", opacity=0.6, size=8),
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Earthquake Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Residuals (Error)", row=1, col=2)

    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        template="plotly_white",
    )
    return fig


def plot_confusion_matrix(results):
    """
    Create a Confusion Matrix by categorizing the continuous predictions
    into 'Low', 'Medium', and 'High' activity levels based on training data quantiles.
    """
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_pred = results["y_test_pred"]

    # Define bins based on training data (33.3% and 66.6% percentiles)
    p33 = np.percentile(y_train, 33.33)
    p67 = np.percentile(y_train, 66.67)

    # Categorize the test and predicted values
    y_test_cat = _categorize_activity(y_test, p33, p67)
    y_pred_cat = _categorize_activity(y_pred, p33, p67)

    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Confusion Matrix (Categorized Activity Levels)",
    )
    fig.update_layout(
        xaxis_title="Predicted Activity Level",
        yaxis_title="Actual Activity Level",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def get_classification_report_df(results):
    """
    Generate a Classification Report DataFrame corresponding to the
    categorized activity levels ('Low', 'Medium', 'High').
    """
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_pred = results["y_test_pred"]

    p33 = np.percentile(y_train, 33.33)
    p67 = np.percentile(y_train, 66.67)

    labels = ["Low", "Medium", "High"]
    report = classification_report(
        _categorize_activity(y_test, p33, p67),
        _categorize_activity(y_pred, p33, p67),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


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
    - **Data**: Recent global earthquake history (monthly or weekly counts)
    - **Method**: Hybrid Model (Exponential Smoothing + Random Forest)
    - **Aggregation**: Monthly counts for stability, weekly counts for finer-grained tests
    - **Train/Test Split**: 80% training, 20% testing
    
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
    - **Pink dotted line**: The n-period future forecast
    - **Gray dots**: Raw aggregated data (noisy)
    
    **Important Limitations:**
    - Earthquake patterns are inherently unpredictable
    - Model captures statistical patterns only, not physical drivers
    
    **Understanding R² Score:**
    - Ranges from 0 to 1 (higher = better fit)
    - Negative: Model worse than simple average (indicative of randomness)
    """
    return explanation


def compare_models(frequency_data, period="M"):
    """
    Compare multiple ML models on the same earthquake frequency data.

    Trains and evaluates 7 different models: Linear Regression, Exponential
    Smoothing, Moving Average, Ridge Regression, ARIMA, Seasonal Naive, and
    Baseline (mean prediction). Returns comparison metrics sorted by Test R².

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns
    period : str
        Aggregation period used to prepare the series ('D', 'W', or 'M')

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

    period = _normalize_period(frequency_data.attrs.get("period", period))

    # Prepare data
    y = frequency_data["count"].values
    split_index = int(len(y) * 0.8)
    y_train = y[:split_index]
    y_test = y[split_index:]

    results_list = []

    # 1. Hybrid Model (Current model from train_ml_model)
    try:
        ml_results = train_ml_model(frequency_data, test_size=0.2, period=period)
        results_list.append(
            {
                "Model": "Hybrid (Exp. Smoothing + RF)",
                "Train R²": ml_results["train_r2"],
                "Test R²": ml_results["test_r2"],
                "Train RMSE": ml_results["train_rmse"],
                "Test RMSE": ml_results["test_rmse"],
                "Train MAE": ml_results["train_mae"],
                "Test MAE": ml_results["test_mae"],
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Hybrid (Exp. Smoothing + RF)"))

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
        exp_train_mae = mean_absolute_error(y_train, exp_train_pred)
        exp_test_mae = mean_absolute_error(y_test, exp_test_pred)
        results_list.append(
            {
                "Model": "Exponential Smoothing",
                "Train R²": exp_train_r2,
                "Test R²": exp_test_r2,
                "Train RMSE": exp_train_rmse,
                "Test RMSE": exp_test_rmse,
                "Train MAE": exp_train_mae,
                "Test MAE": exp_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Exponential Smoothing"))

    # 3. Moving Average (simple implementation for comparison)
    try:
        window_size = 4 if period == "W" else 3
        ma_train = np.convolve(
            y_train, np.ones(window_size) / window_size, mode="valid"
        )
        ma_train_actual = y_train[window_size - 1 :]
        ma_test_pred = np.full(len(y_test), np.mean(y_train))
        ma_train_r2 = r2_score(ma_train_actual, ma_train)
        ma_test_r2 = r2_score(y_test, ma_test_pred)
        ma_train_rmse = np.sqrt(mean_squared_error(ma_train_actual, ma_train))
        ma_test_rmse = np.sqrt(mean_squared_error(y_test, ma_test_pred))
        ma_train_mae = mean_absolute_error(ma_train_actual, ma_train)
        ma_test_mae = mean_absolute_error(y_test, ma_test_pred)
        results_list.append(
            {
                "Model": "Moving Average",
                "Train R²": ma_train_r2,
                "Test R²": ma_test_r2,
                "Train RMSE": ma_train_rmse,
                "Test RMSE": ma_test_rmse,
                "Train MAE": ma_train_mae,
                "Test MAE": ma_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Moving Average"))

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
        ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
        ridge_test_mae = mean_absolute_error(y_test, ridge_test_pred)
        results_list.append(
            {
                "Model": "Ridge Regression",
                "Train R²": ridge_train_r2,
                "Test R²": ridge_test_r2,
                "Train RMSE": ridge_train_rmse,
                "Test RMSE": ridge_test_rmse,
                "Train MAE": ridge_train_mae,
                "Test MAE": ridge_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Ridge Regression"))

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
        arima_train_mae = mean_absolute_error(y_train, arima_train_pred)
        arima_test_mae = mean_absolute_error(y_test, arima_test_pred)
        results_list.append(
            {
                "Model": "ARIMA",
                "Train R²": arima_train_r2,
                "Test R²": arima_test_r2,
                "Train RMSE": arima_train_rmse,
                "Test RMSE": arima_test_rmse,
                "Train MAE": arima_train_mae,
                "Test MAE": arima_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("ARIMA"))

    # 6. Seasonal Naive
    try:
        season = 52 if period == "W" else 12 if period == "M" else 7
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
            sn_train_mae = mean_absolute_error(sn_train_actual, sn_train_pred)
            sn_test_mae = mean_absolute_error(y_test, sn_test_pred)
        else:
            sn_train_r2 = sn_test_r2 = sn_train_rmse = sn_test_rmse = sn_train_mae = (
                sn_test_mae
            ) = 0.0
        results_list.append(
            {
                "Model": "Seasonal Naive",
                "Train R²": sn_train_r2,
                "Test R²": sn_test_r2,
                "Train RMSE": sn_train_rmse,
                "Test RMSE": sn_test_rmse,
                "Train MAE": sn_train_mae,
                "Test MAE": sn_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Seasonal Naive"))

    # 7. Baseline (Mean)
    try:
        baseline_pred_train = np.full(len(y_train), np.mean(y_train))
        baseline_pred_test = np.full(len(y_test), np.mean(y_train))
        baseline_train_r2 = r2_score(y_train, baseline_pred_train)
        baseline_test_r2 = r2_score(y_test, baseline_pred_test)
        baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
        baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred_test))
        baseline_train_mae = mean_absolute_error(y_train, baseline_pred_train)
        baseline_test_mae = mean_absolute_error(y_test, baseline_pred_test)
        results_list.append(
            {
                "Model": "Baseline (Mean)",
                "Train R²": baseline_train_r2,
                "Test R²": baseline_test_r2,
                "Train RMSE": baseline_train_rmse,
                "Test RMSE": baseline_test_rmse,
                "Train MAE": baseline_train_mae,
                "Test MAE": baseline_test_mae,
            }
        )
    except Exception:
        results_list.append(_get_empty_metrics("Baseline (Mean)"))

    # Create dataframe and sort by Test R² (highest first)
    df = pd.DataFrame(results_list)
    df = df.sort_values("Test R²", ascending=False).reset_index(drop=True)
    return df
