"""
Simple Machine Learning module for predicting earthquake frequency trends.

This module uses Linear Regression to predict the number of earthquakes over time.
The goal is to identify trends in earthquake frequency, not to predict exact locations or times.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def prepare_time_series_data(data, period="W"):
    """
    Prepare earthquake frequency data grouped by time period.

    Parameters:
    -----------
    data : pd.DataFrame
        Earthquake data with 'time' column (must be datetime)
    period : str
        Pandas period string ('D' for daily, 'W' for weekly, 'M' for monthly)
        Default is 'W' (weekly)

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'date' and 'count' columns representing earthquake frequency
    """

    # Create a copy to avoid modifying original data
    df = data.copy()

    # Ensure time is datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)

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

    This converts dates to 'days since the earliest date' to make them usable for regression.

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


def train_ml_model(frequency_data, test_size=0.2):
    """
    Train a Linear Regression model to predict earthquake frequency.

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns
    test_size : float
        Fraction of data to use for testing (0.2 = 20%)

    Returns:
    --------
    dict
        Dictionary containing model, scaler, train/test data, and metrics
    """

    # Prepare features (X) and target (y)
    # X: numeric representation of dates
    # y: earthquake frequency (count)
    X, _ = convert_dates_to_numeric(frequency_data["time"])
    y = frequency_data["count"].values

    # Reshape X to 2D array (required by sklearn)
    X = X.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Linear Regression model
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
    }

    return results


def plot_training_process(results):
    """
    Create a matplotlib figure showing training vs test loss over time.

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

    # Plot training and test errors
    epochs = np.arange(len(results["y_train"]))
    train_errors = np.abs(results["y_train"] - results["y_train_pred"])

    ax.scatter(
        results["X_train"] / results["X_train"].max(),
        train_errors,
        alpha=0.6,
        label="Train Error",
        s=30,
    )
    ax.set_xlabel("Time (normalized)", fontsize=11)
    ax.set_ylabel("Absolute Error (earthquake count)", fontsize=11)
    ax.set_title("Training Error Distribution", fontsize=12, fontweight="bold")
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
        X_sorted, y_sorted, "o-", label="Actual", alpha=0.7, linewidth=2, markersize=6
    )
    ax1.plot(
        X_sorted,
        y_pred_sorted,
        "s--",
        label="Predicted",
        alpha=0.7,
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Time (days since start)", fontsize=11)
    ax1.set_ylabel("Earthquake Frequency (count)", fontsize=11)
    ax1.set_title(
        "Earthquake Frequency: Actual vs Predicted", fontsize=12, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot for prediction accuracy
    ax2.scatter(y_all, y_pred_all, alpha=0.6, s=50)

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

    ax2.set_xlabel("Actual Earthquake Frequency", fontsize=11)
    ax2.set_ylabel("Predicted Earthquake Frequency", fontsize=11)
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

    frequency_data = results["frequency_data"]
    model = results["model"]
    scaler = results["scaler"]

    # Prepare data for plotting
    X_all = np.vstack([results["X_train"], results["X_test"]])
    y_pred_all = np.hstack([results["y_train_pred"], results["y_test_pred"]])

    # Get all dates
    all_dates = frequency_data["time"].values
    sorted_idx = np.argsort(X_all.flatten())

    # Generate future predictions
    max_X = X_all.max()
    future_X = np.linspace(max_X + 1, max_X + future_periods, future_periods).reshape(
        -1, 1
    )
    future_X_scaled = scaler.transform(future_X)
    future_y_pred = model.predict(future_X_scaled)

    # Generate future dates (assuming weekly data)
    last_date = pd.to_datetime(all_dates[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=future_periods, freq="W"
    )

    # Create figure
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=frequency_data["count"].values,
            mode="lines+markers",
            name="Actual Frequency",
            line=dict(color="#0ea5a4", width=2),
            marker=dict(size=6),
        )
    )

    # Add predictions on training/test data
    fig.add_trace(
        go.Scatter(
            x=all_dates[sorted_idx],
            y=y_pred_all[sorted_idx],
            mode="lines",
            name="Model Predictions (Historical)",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )

    # Add future predictions
    fig.add_trace(
        go.Scatter(
            x=list(all_dates) + list(future_dates),
            y=list(y_pred_all[sorted_idx]) + list(future_y_pred),
            mode="lines",
            name="Future Trend (Extrapolated)",
            line=dict(color="#ec4899", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Earthquake Frequency Trend: Actual vs Predicted",
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
    
    **What it does:**
    This model predicts earthquake **frequency trends** (how many earthquakes occur over time).
    It does NOT predict:
    - Exact earthquake locations
    - Exact earthquake times
    - Specific earthquake magnitudes
    
    **How it works:**
    1. **Data Preparation**: Earthquake data is grouped by week, counting how many occur each week
    2. **Date Conversion**: Dates are converted to numbers (days since the first date)
    3. **Model Training**: A Linear Regression model learns the relationship between time and frequency
    4. **Prediction**: The model finds the trend and predicts future frequency patterns
    
    **Why Linear Regression?**
    - Simple and interpretable
    - Good for understanding overall trends
    - Shows if earthquake frequency is increasing or decreasing
    
    **Limitations:**
    - Real earthquake patterns are complex and chaotic
    - This model only captures general trends, not short-term variations
    - External factors (e.g., tectonic changes) are not accounted for
    - Historical data quality varies by region
    
    **Model Score (R² Score):**
    - Ranges from 0 to 1 (higher is better)
    - 1.0 = perfect predictions
    - 0.5 = model explains 50% of the variation
    - 0.0 = model is no better than just guessing the average
    """

    return explanation
