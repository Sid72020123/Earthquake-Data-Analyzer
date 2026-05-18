"""
Machine Learning module for predicting earthquake frequency trends.

This module uses Polynomial Regression with data smoothing to predict
earthquake frequency trends. The model captures non-linear patterns in
earthquake occurrence rates over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def get_ml_data_from_full_history(data, years=3):
    """
    Extract ML training data from full earthquake history (last N years).

    Uses the full historical dataset instead of filtered dashboard data,
    ensuring consistent ML training regardless of dashboard filters.

    Parameters:
    -----------
    data : pd.DataFrame
        Full earthquake data with 'time' column
    years : int
        Number of years of historical data to use (default: 3)

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
    Prepare earthquake frequency data grouped by time period with smoothing.

    Monthly aggregation (instead of weekly) provides more stable trends
    while reducing noise from short-term fluctuations.

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
        DataFrame with 'time', 'count', and 'count_smoothed' columns
    """

    # Create a copy to avoid modifying original data
    df = data.copy()

    # Ensure time is datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Convert to timezone-naive to avoid UserWarning with to_period()
    df["time"] = df["time"].dt.tz_localize(None)

    # Group by period and count earthquakes
    frequency_data = (
        df.groupby(df["time"].dt.to_period(period)).size().reset_index(name="count")
    )

    # Convert period to timestamp (start of period)
    frequency_data["time"] = frequency_data["time"].dt.to_timestamp()
    frequency_data = frequency_data.sort_values("time").reset_index(drop=True)

    # Apply smoothing using exponential weighted moving average to reduce noise
    # This captures trends while filtering out random spikes
    frequency_data["count_smoothed"] = (
        frequency_data["count"].ewm(span=3, adjust=False).mean()
    )

    return frequency_data


def convert_dates_to_numeric(dates):
    """
    Convert datetime values to numeric values for ML model.

    This converts dates to 'days since the earliest date' for regression.

    Parameters:
    -----------
    dates : pd.Series or np.ndarray
        Datetime values to convert

    Returns:
    --------
    np.ndarray
        Numeric representation of dates (days since minimum date)
    int
        The minimum date timestamp (for reference)
    """

    # Convert to datetime if not already
    dates = pd.to_datetime(dates)

    # Get the minimum date as reference
    min_date = dates.min()

    # Convert to days since minimum date
    numeric_dates = (dates - min_date).dt.total_seconds() / (24 * 3600)

    return numeric_dates.values, min_date.timestamp()


def train_ml_model(frequency_data, test_size=0.2, poly_degree=2):
    """
    Train a Polynomial Regression model to predict earthquake frequency.

    Uses polynomial features (degree 2) with data smoothing to better
    capture non-linear trends in earthquake frequency.

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time', 'count', and 'count_smoothed' columns
    test_size : float
        Fraction of data to use for testing (0.2 = 20%)
    poly_degree : int
        Degree of polynomial features (default: 2)

    Returns:
    --------
    dict
        Dictionary containing model, scaler, train/test data, and metrics
    """

    # Prepare features (X) and target (y)
    # X: numeric representation of dates
    # y: earthquake frequency using smoothed data (less noisy)
    X, _ = convert_dates_to_numeric(frequency_data["time"])
    y = frequency_data["count_smoothed"].values

    # Reshape X to 2D array (required by sklearn)
    X = X.reshape(-1, 1)

    # Split data chronologically so the model respects the time-series order.
    split_index = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_test) == 0:
        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]

    # Create polynomial features and scale them
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale features for better numerical stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Train the Polynomial Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    # Store results in a dictionary
    results = {
        "model": model,
        "scaler": scaler,
        "poly": poly,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "frequency_data": frequency_data,
        "poly_degree": poly_degree,
    }

    return results


def plot_training_process(results):
    """
    Create a matplotlib figure showing error distribution during training.

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with training visualization
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot training errors
    train_errors = np.abs(results["y_train"] - results["y_train_pred"])
    ax.scatter(
        results["X_train"] / results["X_train"].max(),
        train_errors,
        alpha=0.6,
        label="Train Error",
        s=30,
        color="#0ea5a4",
    )

    ax.set_xlabel("Time (normalized)", fontsize=11)
    ax.set_ylabel("Absolute Error (earthquake count)", fontsize=11)
    ax.set_title("Model Error Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(results):
    """
    Create a matplotlib figure showing actual vs predicted earthquake frequency.

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

    # Convert back to original time values for x-axis
    X_all = np.vstack([results["X_train"], results["X_test"]])
    y_all = np.hstack([results["y_train"], results["y_test"]])
    y_pred_all = np.hstack([results["y_train_pred"], results["y_test_pred"]])

    # Plot 1: Time series comparison
    sorted_idx = np.argsort(X_all.flatten())
    X_sorted = X_all[sorted_idx].flatten()
    y_sorted = y_all[sorted_idx]
    y_pred_sorted = y_pred_all[sorted_idx]

    ax1.plot(
        X_sorted,
        y_sorted,
        "o-",
        label="Actual (Smoothed)",
        alpha=0.7,
        linewidth=2,
        markersize=6,
        color="#0ea5a4",
    )
    ax1.plot(
        X_sorted,
        y_pred_sorted,
        "s--",
        label="Predicted",
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

    # Add perfect prediction line
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
    Create a Plotly interactive figure with predictions and trend line.

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()
    future_periods : int
        Number of future periods to predict (default: 12)

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """

    frequency_data = results["frequency_data"].copy()
    model = results["model"]
    scaler = results["scaler"]
    poly = results["poly"]

    # Prepare dates and a shared numeric reference so historical and future
    # predictions stay aligned with the monthly timeline.
    frequency_data["time"] = pd.to_datetime(frequency_data["time"])
    all_dates = pd.to_datetime(frequency_data["time"])
    all_numeric_dates, min_date_timestamp = convert_dates_to_numeric(all_dates)
    all_numeric_dates = all_numeric_dates.reshape(-1, 1)
    min_date = pd.to_datetime(min_date_timestamp, unit="s")

    historical_scaled = scaler.transform(poly.transform(all_numeric_dates))
    historical_predictions = model.predict(historical_scaled)

    # Generate future predictions
    future_dates = pd.date_range(
        start=all_dates.iloc[-1] + pd.DateOffset(months=1),
        periods=future_periods,
        freq="MS",
    )
    future_numeric_dates = np.array(
        [(date - min_date).total_seconds() / (24 * 3600) for date in future_dates]
    ).reshape(-1, 1)
    future_X_poly = poly.transform(future_numeric_dates)
    future_X_scaled = scaler.transform(future_X_poly)
    future_y_pred = model.predict(future_X_scaled)

    # Create figure
    fig = go.Figure()

    # Add actual raw data (lightly)
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=frequency_data["count"].values,
            mode="markers",
            name="Raw Data (Noisy)",
            marker=dict(size=6, color="#94a3b8"),
            opacity=0.5,
        )
    )

    # Add smoothed actual data
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=frequency_data["count_smoothed"].values,
            mode="lines",
            name="Actual Trend (Smoothed)",
            line=dict(color="#0ea5a4", width=3),
        )
    )

    # Add fitted model values across the full historical timeline.
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=historical_predictions,
            mode="lines",
            name="Model Predictions",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )

    # Add future predictions
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
        title="Earthquake Frequency Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Earthquake Frequency (count)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        font=dict(size=11),
    )

    return fig


def get_model_explanation():
    """
    Return a beginner-friendly explanation of the ML model.

    Returns:
    --------
    str
        Explanation text in markdown format
    """

    explanation = """
    ### 🤖 About This ML Model
    
    **What it predicts:**
    Earthquake **frequency trends** over time (is activity increasing or decreasing?).
    
    **What it does NOT predict:**
    - Exact earthquake locations or times
    - Specific earthquake magnitudes  
    - Earthquake timing accuracy
    
    **Key Features:**
    - **Data**: Last 3 years of global earthquake data (~36 months)
    - **Method**: Polynomial Regression (degree 2) with data smoothing
    - **Smoothing**: Exponential weighted moving average reduces noise
    - **Monthly Aggregation**: More stable than weekly data
    - **Training**: 80% training, 20% test data
    
    **Why this approach works:**
    - Earthquake frequency is noisy and non-linear
    - Polynomial features capture curved trends better than straight lines
    - Data smoothing filters out random spikes and fluctuations
    - Monthly grouping provides stable, meaningful patterns
    
    **Important Limitations:**
    - Earthquake patterns are inherently chaotic
    - Model captures general trends only, not short-term changes
    - Historical data quality varies by region
    - External factors (tectonic shifts, climate, instrumentation) not modeled
    
    **Understanding R² Score:**
    - Ranges from 0 to 1 (higher = better fit)
    - 0.5-0.7: Reasonable trend capture
    - 0.3-0.5: Captures some patterns
    - Below 0.3: High randomness dominates
    """

    return explanation
