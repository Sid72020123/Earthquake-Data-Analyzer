"""
Machine Learning module for predicting earthquake frequency trends.

This module uses Random Forest Regressor to forecast earthquake frequency trends
over time. Random Forest is ideal for earthquake data because it handles non-linear
patterns and noisy data well, making it better than simple linear approaches for
capturing complex earthquake behavior patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
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


def train_ml_model(frequency_data, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regressor to predict earthquake frequency.

    Random Forest is a powerful machine learning algorithm that:
    - Handles non-linear patterns well (earthquake data is non-linear)
    - Works with noisy data (earthquake patterns are chaotic)
    - Can capture complex relationships without overfitting
    - Provides feature importance insights

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns
    test_size : float
        Fraction of data to use for testing (0.2 = 20%, 0.8 = 80% training)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing model, data, and performance metrics
    """
    # Prepare features (X) and target (y)
    # X: numeric representation of dates (time)
    # y: earthquake frequency count
    X, min_date_timestamp = convert_dates_to_numeric(frequency_data["time"])
    y = frequency_data["count"].values

    # Reshape X to 2D array (required by sklearn models)
    X = X.reshape(-1, 1)

    # Split data chronologically - keep time order so the model
    # learns temporal patterns correctly
    split_index = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Safety check: ensure we have test data
    if len(X_test) == 0:
        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]

    # Create and train Random Forest Regressor
    # Simple parameters for beginner-friendly implementation
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        max_depth=10,  # Maximum depth of each tree
        min_samples_split=5,  # Minimum samples to split a node
        min_samples_leaf=2,  # Minimum samples in leaf node
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores for speed
    )

    # Fit the model to training data
    model.fit(X_train, y_train)

    # Make predictions on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Calculate feature importance (time is the only feature)
    feature_importance = model.feature_importances_[0]

    # Store results in a dictionary for easy access
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
        "frequency_data": frequency_data,
        "feature_importance": feature_importance,
        "min_date_timestamp": min_date_timestamp,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_total_samples": len(X_train) + len(X_test),
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
        label="Random Forest Predictions",
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
    min_date_timestamp = results["min_date_timestamp"]

    # Prepare historical data
    frequency_data["time"] = pd.to_datetime(frequency_data["time"])
    all_dates = pd.to_datetime(frequency_data["time"])
    all_numeric_dates, _ = convert_dates_to_numeric(all_dates)
    all_numeric_dates = all_numeric_dates.reshape(-1, 1)
    min_date = pd.to_datetime(min_date_timestamp, unit="s")

    # Make predictions for historical data
    historical_predictions = model.predict(all_numeric_dates)

    # Generate future predictions
    last_date = all_dates.iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_periods,
        freq="MS",
    )
    future_numeric_dates = np.array(
        [(date - min_date).total_seconds() / (24 * 3600) for date in future_dates]
    ).reshape(-1, 1)
    future_y_pred = model.predict(future_numeric_dates)

    # Create interactive figure
    fig = go.Figure()

    # Add raw data (lightly - to show it's noisy)
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

    # Add model predictions for historical period
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=historical_predictions,
            mode="lines",
            name="Random Forest Trend",
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

    For Random Forest with time data, this shows how important
    time is for predictions (usually 1.0 = very important).

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
    ax.set_title(
        "Feature Importance in Random Forest Model", fontsize=12, fontweight="bold"
    )
    ax.set_xlim(0, 1.1)

    # Add value label on bar
    ax.text(importance + 0.02, 0, f"{importance:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    return fig


def get_model_explanation():
    """
    Return a beginner-friendly explanation of the Random Forest ML model.

    Returns:
    --------
    str
        Explanation text in markdown format
    """
    explanation = """
    ### 🤖 About This Random Forest Model
    
    **What it predicts:**
    Earthquake **frequency trends** over time (is activity increasing or decreasing?).
    
    **What it does NOT predict:**
    - Exact earthquake locations or times
    - Specific earthquake magnitudes  
    - When earthquakes will occur
    
    **Key Features:**
    - **Data**: Last 3 years of global earthquake data (~36 months)
    - **Method**: Random Forest Regressor (100 decision trees)
    - **Aggregation**: Monthly earthquake counts for stability
    - **Train/Test Split**: 80% training, 20% testing
    
    **Why Random Forest works well for earthquakes:**
    - Earthquake patterns are **non-linear** (not a straight line)
    - Earthquake data is **noisy** (random fluctuations)
    - Random Forest handles both very well by building many decision trees
    - Each tree learns different patterns, and averaging them gives smooth predictions
    - No complex scaling or preprocessing needed - Random Forest handles this naturally
    
    **How it works (simplified):**
    1. Build 100 different decision trees using random data samples
    2. Each tree learns "if time > X then earthquake count ≈ Y"
    3. Average predictions from all 100 trees to get final prediction
    4. This averaging reduces noise and captures complex patterns
    
    **Important Limitations:**
    - Earthquake patterns are inherently chaotic and unpredictable
    - Model captures general trends only, not short-term changes
    - Historical data quality varies by region and time period
    - External factors (tectonic shifts, climate, etc.) not in the model
    - Small data sample (36 months) may not capture all patterns
    
    **Understanding R² Score:**
    - Ranges from 0 to 1 (higher = better fit)
    - 0.5-1.0: Model captures meaningful patterns
    - 0.3-0.5: Model captures some patterns
    - Below 0.3: Earthquake randomness dominates, predictions unreliable
    - Negative: Model worse than simple average (very rare)
    """
    return explanation
