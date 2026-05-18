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
    Train a simple Moving Average model to predict earthquake frequency.

    The Moving Average is a robust baseline for noisy time series like
    earthquake monthly counts. It avoids overfitting and is easy to explain.

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

    # Compute moving average predictions for training (use past values only)
    train_series = pd.Series(y_train)
    # Prediction at t is mean of previous `window` actuals (no lookahead)
    y_train_pred = (
        train_series.shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
        .fillna(train_series.mean())
        .values
    )

    # Forecasting for test: iterative forecast using last `window` values (use predictions for future steps)
    history = list(y_train)
    y_test_pred = []
    for _ in range(len(y_test)):
        pred = (
            float(np.mean(history[-window:]))
            if len(history) > 0
            else float(np.mean(y_train))
        )
        y_test_pred.append(pred)
        # use prediction to extend history (no future leakage)
        history.append(pred)
    y_test_pred = np.array(y_test_pred)

    # Simple model object with forecast method for compatibility
    class MovingAverageModel:
        def __init__(self, window, history):
            self.window = int(window)
            self.history = list(history)

        def forecast(self, steps=1):
            preds = []
            h = list(self.history)
            for _ in range(int(steps)):
                p = float(np.mean(h[-self.window :])) if len(h) > 0 else 0.0
                preds.append(p)
                h.append(p)
            return np.array(preds)

    model = MovingAverageModel(window=window, history=history)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

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
        "feature_importance": 1.0,
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
        label="Moving Average Trend",
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
            name="Moving Average Trend",
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
    fig, ax = plt.subplots(figsize=(8, 4))

    importance = results["feature_importance"]
    ax.barh(["Time"], [importance], color="#0ea5a4")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance (Time)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.1)

    # Add value label on bar
    ax.text(importance + 0.02, 0, f"{importance:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    return fig


def get_model_explanation():
    """
    Return a beginner-friendly explanation of the Moving Average model.

    Returns:
    --------
    str
        Explanation text in markdown format
    """
    explanation = """
    ### 🤖 About This Moving Average Model
    
    **What it predicts:**
    Earthquake **frequency trends** over time using short-window smoothing.
    
    **What it does NOT predict:**
    - Exact earthquake locations or times
    - Specific earthquake magnitudes  
    - When earthquakes will occur
    
    **Key Features:**
    - **Data**: Last 5 years of global earthquake data (~60 months)
    - **Method**: Moving Average (simple smoothing)
    - **Aggregation**: Monthly earthquake counts for stability
    - **Train/Test Split**: 80% training (~48 months), 20% testing (~12 months)
    - **Best for**: Noisy time series where short-term smoothing helps
    
    **Why Moving Average is appropriate:**
    - Earthquake frequency is **highly chaotic and unpredictable**
    - Moving Average smooths short-term noise and reveals local direction
    - Avoids overfitting compared to complex models
    - Simple and interpretable for beginners
    
    **How it works (simplified):**
    1. Load 5 years (~60 months) of global earthquake data
    2. Count earthquakes by month for stability
    3. Split into 80% training (~48 months) and 20% testing (~12 months)
    4. Compute moving average using a short window (default 3 months)
    5. Use the moving average as the prediction for the next periods
    
    **What the model shows:**
    - **Teal line**: The smoothed moving-average trend
    - **Pink dotted line**: The n-month forecast (repeats recent average)
    - **Gray dots**: Raw monthly data (noisy)
    
    **Important Limitations:**
    - Earthquake patterns are inherently chaotic and unpredictable
    - Model captures short-term smoothing only, not causal drivers
    - Historical data quality varies by region and time period
    - Assumes recent averages are informative for the near future
    
    **Understanding R² Score:**
    - Ranges from 0 to 1 (higher = better fit)
    - Negative: Model worse than simple average (indicative of randomness)
    """
    return explanation
