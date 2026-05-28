"""
Machine Learning module for predicting earthquake frequency trends.

This module uses a Hybrid Model (Exponential Smoothing + Gradient Boosting) to
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


def _build_time_features(dates, period, start_idx=0, y_history=None, n_lags=3):
    """
    Build time-based and lag features for the residual ML model.

    Lag features capture autocorrelation in earthquake frequency: last month's
    count is often the best predictor of this month's count. Rolling means
    smooth out individual spike noise.

    Parameters:
    -----------
    dates : array-like
        Timestamps for each row.
    period : str
        Aggregation period ('D', 'W', or 'M').
    start_idx : int
        Index offset used as the time_idx feature.
    y_history : array-like or None
        Full count series up to (and including) the training window, used
        to build look-back lag features. Pass None to skip lag features.
    n_lags : int
        Number of lag periods to include.
    """
    period = _normalize_period(period)
    date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    n = len(date_series)

    features = pd.DataFrame(
        {
            "time_idx": np.arange(start_idx, start_idx + n),
            "month": date_series.dt.month,
            "quarter": date_series.dt.quarter,
        }
    )

    if period == "W":
        features["week_of_year"] = date_series.dt.isocalendar().week.astype(int)
    elif period == "D":
        features["day_of_year"] = date_series.dt.dayofyear

    # Lag + rolling-mean features (require y_history)
    if y_history is not None:
        hist = np.asarray(y_history, dtype=float)
        train_mean = float(np.mean(hist))
        for lag in range(1, n_lags + 1):
            vals = []
            for i in range(n):
                # index into the history array: position just before this row
                pos = len(hist) - n + i - lag
                vals.append(float(hist[pos]) if 0 <= pos < len(hist) else train_mean)
            features[f"lag_{lag}"] = vals
        # 3-period rolling mean of the most recent actual values
        roll3 = []
        for i in range(n):
            idxs = [len(hist) - n + i - k for k in range(1, 4)]
            valid = [hist[j] for j in idxs if 0 <= j < len(hist)]
            roll3.append(float(np.mean(valid)) if valid else train_mean)
        features["roll_mean_3"] = roll3

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
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce", utc=True)

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
    short-term fluctuations. Partial periods at the boundaries (e.g. the
    current month, or a start month with only a few days of data) are
    automatically dropped so they do not bias the model.

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
        DataFrame with 'time' and 'count' columns, partial boundary periods removed
    """
    period = _normalize_period(period)

    df = data.copy()
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.tz_localize(None)

    frequency_data = (
        df.groupby(df["time"].dt.to_period(period)).size().reset_index(name="count")
    )
    frequency_data["time"] = frequency_data["time"].dt.to_timestamp()
    frequency_data = frequency_data.sort_values("time").reset_index(drop=True)
    frequency_data.attrs["period"] = period

    # Drop partial boundary periods: a period is considered partial when its
    # count is less than 60% of the median count.  We only trim from the
    # very start and very end of the series (not interior spikes).
    if len(frequency_data) > 4:
        median_count = frequency_data["count"].median()
        threshold = 0.60 * median_count
        while len(frequency_data) > 4 and frequency_data["count"].iloc[0] < threshold:
            frequency_data = frequency_data.iloc[1:].reset_index(drop=True)
        while len(frequency_data) > 4 and frequency_data["count"].iloc[-1] < threshold:
            frequency_data = frequency_data.iloc[:-1].reset_index(drop=True)
        frequency_data.attrs["period"] = period

    return frequency_data


def train_ml_model(
    frequency_data, test_size=0.2, window=3, random_state=42, period="M"
):
    """
    Train a Hybrid Model (Exponential Smoothing + Gradient Boosting) to predict
    earthquake frequency trends.

    Improvements over a plain Exponential Smoothing or standard machine learning approach:
    - Gradient Boosting with 'huber' loss is robust to spike outliers (large
      earthquakes cause temporary spikes that would otherwise dominate MSE).
    - Lag features let the ML component exploit autocorrelation in the counts.
    - Rolling-mean features smooth short-term noise.
    - Seasonal ES is tried first when enough data is present.

    Parameters:
    -----------
    frequency_data : pd.DataFrame
        Time series data with 'time' and 'count' columns
    test_size : float
        Fraction of data to use for testing (default 0.2)
    window : int
        Unused; kept for API compatibility.
    random_state : int
        Random seed for reproducibility.
    period : str
        Aggregation period used to build the input series ('D', 'W', or 'M')

    Returns:
    --------
    dict
        Dictionary containing model, data, and performance metrics
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    period = _normalize_period(frequency_data.attrs.get("period", period))

    dates = pd.to_datetime(frequency_data["time"])
    y = frequency_data["count"].values.astype(float)

    # Chronological split
    split_index = max(1, int(len(y) * (1 - test_size)))
    y_train = y[:split_index]
    y_test = y[split_index:]
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    # Safety check — ensure at least one test sample
    if len(y_test) == 0:
        y_train = y[:-1]
        y_test = y[-1:]
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    split_index = len(y_train)

    # ── Base Model: Exponential Smoothing ─────────────────────────────────────
    # Try additive seasonal (period=12 or 52) when enough data is available.
    # Fall back to non-seasonal, then to the training mean.
    season_map = {"M": 12, "W": 52, "D": 7}
    season_len = season_map.get(period, 12)
    base_model = None
    try:
        if len(y_train) >= season_len * 2:
            base_model = ExponentialSmoothing(
                y_train,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=season_len,
                initialization_method="estimated",
            ).fit(optimized=True)
        else:
            raise ValueError("Not enough data for seasonal ES")
    except Exception:
        try:
            base_model = ExponentialSmoothing(
                y_train,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            base_model = None

    if base_model is not None:
        base_train_pred = base_model.fittedvalues
        base_test_pred = base_model.forecast(steps=len(y_test))
    else:
        base_train_pred = np.full(len(y_train), float(np.mean(y_train)))
        base_test_pred = np.full(len(y_test), float(np.mean(y_train)))

    residuals_train = y_train - base_train_pred

    # ── Feature Engineering ───────────────────────────────────────────────────
    # Build lag + rolling features using actual y values so the model can
    # leverage autocorrelation.  For the test window we pass the full y array
    # so lags into the training period are always real observed values.
    N_LAGS = 3
    X_train_ml = _build_time_features(
        dates.iloc[:split_index],
        period,
        start_idx=0,
        y_history=y_train,
        n_lags=N_LAGS,
    )
    X_test_ml = _build_time_features(
        dates.iloc[split_index:],
        period,
        start_idx=split_index,
        y_history=y,
        n_lags=N_LAGS,  # full y so test lags see real train values
    )

    # ── Residual Model: Gradient Boosting (huber loss) ────────────────────────
    # Huber loss down-weights large residuals (spikes) and is far more robust
    # than standard squared-error models for noisy earthquake data.
    # Conservative settings prevent the GBM from memorising training residuals
    # (which would hurt test performance on unseen spike events).
    gb_model = GradientBoostingRegressor(
        n_estimators=25,
        max_depth=1,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=4,
        loss="huber",
        alpha=0.9,  # huber quantile — focus on the 90th-percentile boundary
        random_state=random_state,
    )
    gb_model.fit(X_train_ml, residuals_train)

    gb_train_pred = gb_model.predict(X_train_ml)
    gb_test_pred = gb_model.predict(X_test_ml)

    # ── Final Predictions ─────────────────────────────────────────────────────
    y_train_pred = np.maximum(base_train_pred + gb_train_pred, 0)
    y_test_pred = np.maximum(base_test_pred + gb_test_pred, 0)

    # ── Forecast wrapper ──────────────────────────────────────────────────────
    # The inner class stores the full y array so future-date lags always
    # resolve to real observed counts rather than zeros or training means.
    class HybridModel:
        def __init__(self, base, gb, last_date, period, full_y, y_train_mean):
            self.base = base
            self.gb = gb
            self.last_date = pd.to_datetime(last_date)
            self.period = period
            self._full_y = np.asarray(full_y, dtype=float)
            self.y_train_mean = y_train_mean

        def forecast(self, steps=1):
            if self.base is not None:
                try:
                    base_f = self.base.forecast(steps=steps)
                except Exception:
                    base_f = np.full(steps, self.y_train_mean)
            else:
                base_f = np.full(steps, self.y_train_mean)

            future_dates = _get_future_dates(self.last_date, steps, self.period)
            # For future lags we extend the known history with the base forecast
            extended_y = np.concatenate([self._full_y, base_f])

            X_f_ml = _build_time_features(
                future_dates,
                self.period,
                start_idx=len(self._full_y),
                y_history=extended_y,
                n_lags=N_LAGS,
            )
            gb_f = self.gb.predict(X_f_ml)
            return np.maximum(base_f + gb_f, 0)

    model = HybridModel(
        base_model, gb_model, dates.iloc[-1], period, y, np.mean(y_train)
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 1))) * 100
    test_max_error = float(np.max(np.abs(y_test - y_test_pred)))

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
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_total_samples": len(y_train) + len(y_test),
    }

    return results


def plot_actual_vs_predicted(results):
    """
    Create a Plotly scatter plot showing actual vs predicted earthquake frequency.

    Shows Prediction accuracy (points near diagonal = good predictions)

    Parameters:
    -----------
    results : dict
        Results dictionary from train_ml_model()

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with actual vs predicted comparison
    """
    fig = go.Figure()

    # Combine all data for visualization
    y_all = np.hstack([results["y_train"], results["y_test"]])
    y_pred_all = np.hstack([results["y_train_pred"], results["y_test_pred"]])

    # Plot Scatter plot for prediction accuracy
    fig.add_trace(
        go.Scatter(
            x=y_all,
            y=y_pred_all,
            mode="markers",
            name="Predictions",
            marker=dict(color="#0ea5a4", opacity=0.6, size=8),
        )
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
        )
    )

    fig.update_xaxes(title_text="Actual Earthquake Frequency")
    fig.update_yaxes(title_text="Predicted Earthquake Frequency")

    fig.update_layout(
        title="Prediction Accuracy (Actual vs Predicted)",
        height=450,
        showlegend=True,
        margin=dict(t=50, b=20, l=20, r=20),
        template="plotly_white",
    )
    return fig


def create_prediction_plotly(results, future_periods=12, display_history_periods=None):
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

    # Filter historical data for visualization if requested
    if display_history_periods is not None and display_history_periods < len(all_dates):
        all_dates = all_dates.iloc[-display_history_periods:]
        historical_predictions = historical_predictions[-display_history_periods:]
        actual_counts = actual_counts[-display_history_periods:]

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
            x=[all_dates.iloc[-1]] + list(future_dates),
            y=[historical_predictions[-1]] + list(future_y_pred),
            mode="lines",
            name="Future Forecast",
            line=dict(color="#ec4899", width=2, dash="dot"),
        )
    )

    # Fix x-axis formatting for Weekly data so it doesn't show just months
    if period == "W":
        fig.update_xaxes(tickformat="%Y-%m-%d", dtick=604800000)  # 7 days in ms

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


def plot_confusion_matrix(results):
    """
    Create a Confusion Matrix by categorizing the continuous predictions
    into 'Low', 'Medium', and 'High' activity levels based on training data quantiles.
    """
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_pred = results["y_test_pred"]

    # Use slightly wider bins to reduce class-boundary noise on small test windows
    p33 = np.percentile(y_train, 30.0)
    p67 = np.percentile(y_train, 70.0)

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

    p33 = np.percentile(y_train, 30.0)
    p67 = np.percentile(y_train, 70.0)

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
    - **Data**: Full global earthquake history (monthly or weekly counts)
    - **Method**: Hybrid Model (Exponential Smoothing + Gradient Boosting)
    - **Aggregation**: Monthly counts for stability, weekly counts for finer-grained tests
    - **Train/Test Split**: 80% training, 20% testing
    
    **Why a Hybrid Model?**
    - Earthquake frequency is **highly chaotic and noisy**
    - **Exponential Smoothing** (with optional seasonal component) captures the underlying trend
    - **Gradient Boosting** with Huber loss learns from residuals — it is robust to spike outliers
      (large earthquake swarms) that would otherwise dominate mean-squared-error models
    - **Lag features** let the model exploit autocorrelation (last month's count helps predict this month)
    - Combining them yields higher accuracy than any single model
    
    **How it works (simplified):**
    1. Base model (ES) calculates the general trend of earthquake frequency.
    2. We measure the "errors" (residuals) between actual data and the base trend.
    3. Gradient Boosting predicts these errors using time + lag features.
    4. Final prediction = Base Trend + Gradient Boosting Adjustment.
    
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
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
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

    def _compute_metrics(name, train_actual, train_pred, test_actual, test_pred):
        test_mape = (
            np.mean(np.abs((test_actual - test_pred) / np.maximum(test_actual, 1)))
            * 100
        )
        test_max_error = float(np.max(np.abs(test_actual - test_pred)))
        return {
            "Model": name,
            "Accuracy (%)": max(0.0, 100.0 - test_mape),
            "Train R²": r2_score(train_actual, train_pred),
            "Test R²": r2_score(test_actual, test_pred),
            "Train RMSE": np.sqrt(mean_squared_error(train_actual, train_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(test_actual, test_pred)),
            "Train MAE": mean_absolute_error(train_actual, train_pred),
            "Test MAE": mean_absolute_error(test_actual, test_pred),
            "Test MAPE": test_mape,
            "Test Max Error": test_max_error,
        }

    def _get_empty_metrics_ext(name):
        return {
            "Model": name,
            "Accuracy (%)": 0.0,
            "Train R²": 0.0,
            "Test R²": 0.0,
            "Train RMSE": 0.0,
            "Test RMSE": 0.0,
            "Train MAE": 0.0,
            "Test MAE": 0.0,
            "Test MAPE": 0.0,
            "Test Max Error": 0.0,
        }

    # 1. Hybrid Model (current model from train_ml_model)
    try:
        ml_results = train_ml_model(frequency_data, test_size=0.2, period=period)
        results_list.append(
            _compute_metrics(
                "Hybrid (ES + GBM)",
                ml_results["y_train"],
                ml_results["y_train_pred"],
                ml_results["y_test"],
                ml_results["y_test_pred"],
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Hybrid (ES + GBM)"))

    # 2. Exponential Smoothing
    try:
        exp_smooth = ExponentialSmoothing(
            y_train, trend="add", seasonal=None, initialization_method="estimated"
        )
        exp_model = exp_smooth.fit(optimized=True)
        exp_train_pred = exp_model.fittedvalues
        exp_test_pred = exp_model.forecast(steps=len(y_test))
        results_list.append(
            _compute_metrics(
                "Exponential Smoothing", y_train, exp_train_pred, y_test, exp_test_pred
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Exponential Smoothing"))

    # 3. Moving Average (simple implementation for comparison)
    try:
        window_size = 4 if period == "W" else 3
        ma_train = np.convolve(
            y_train, np.ones(window_size) / window_size, mode="valid"
        )
        ma_train_actual = y_train[window_size - 1 :]
        ma_test_pred = np.full(len(y_test), np.mean(y_train))
        results_list.append(
            _compute_metrics(
                "Moving Average", ma_train_actual, ma_train, y_test, ma_test_pred
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Moving Average"))

    # 4. Ridge Regression
    try:
        X_train_idx = np.arange(len(y_train)).reshape(-1, 1)
        X_test_idx = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_idx, y_train)
        ridge_train_pred = ridge_model.predict(X_train_idx)
        ridge_test_pred = ridge_model.predict(X_test_idx)
        results_list.append(
            _compute_metrics(
                "Ridge Regression", y_train, ridge_train_pred, y_test, ridge_test_pred
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Ridge Regression"))

    # 5. ARIMA
    try:
        if len(y_train) > 300:
            raise ValueError("Too much data for ARIMA, skipping to prevent hanging.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Non-invertible starting MA parameters found.*",
                category=UserWarning,
            )
            arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
        arima_train_pred = arima_model.fittedvalues
        arima_test_pred = arima_model.get_forecast(steps=len(y_test)).predicted_mean
        results_list.append(
            _compute_metrics(
                "ARIMA", y_train, arima_train_pred, y_test, arima_test_pred
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("ARIMA"))

    # 6. Seasonal Naive
    try:
        season = 52 if period == "W" else 12 if period == "M" else 7
        if len(y_train) >= season:
            sn_train_actual = y_train[season:]
            sn_train_pred = y_train[:-season]  # lag by one season
            sn_test_pred = np.full(
                len(y_test),
                y_train[-season] if len(y_train) >= season else np.mean(y_train),
            )
            results_list.append(
                _compute_metrics(
                    "Seasonal Naive",
                    sn_train_actual,
                    sn_train_pred,
                    y_test,
                    sn_test_pred,
                )
            )
        else:
            results_list.append(_get_empty_metrics_ext("Seasonal Naive"))
    except Exception:
        results_list.append(_get_empty_metrics_ext("Seasonal Naive"))

    # 7. Baseline (Mean)
    try:
        baseline_pred_train = np.full(len(y_train), np.mean(y_train))
        baseline_pred_test = np.full(len(y_test), np.mean(y_train))
        results_list.append(
            _compute_metrics(
                "Baseline (Mean)",
                y_train,
                baseline_pred_train,
                y_test,
                baseline_pred_test,
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Baseline (Mean)"))

    # 8. Gradient Boosting (Hybrid Base)
    try:
        X_train_idx = np.arange(len(y_train)).reshape(-1, 1)
        X_test_idx = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)
        gbm_model = GradientBoostingRegressor(random_state=42)
        gbm_model.fit(X_train_idx, y_train)
        gbm_train_pred = gbm_model.predict(X_train_idx)
        gbm_test_pred = gbm_model.predict(X_test_idx)
        results_list.append(
            _compute_metrics(
                "Gradient Boosting (Hybrid Base)",
                y_train,
                gbm_train_pred,
                y_test,
                gbm_test_pred,
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Gradient Boosting (Hybrid Base)"))

    # 9. Random Forest
    try:
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_idx, y_train)
        rf_train_pred = rf_model.predict(X_train_idx)
        rf_test_pred = rf_model.predict(X_test_idx)
        results_list.append(
            _compute_metrics(
                "Random Forest", y_train, rf_train_pred, y_test, rf_test_pred
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Random Forest"))

    # 10. Support Vector Regression
    try:
        svr_model = SVR()
        svr_model.fit(X_train_idx, y_train)
        svr_train_pred = svr_model.predict(X_train_idx)
        svr_test_pred = svr_model.predict(X_test_idx)
        results_list.append(
            _compute_metrics(
                "Support Vector Regression",
                y_train,
                svr_train_pred,
                y_test,
                svr_test_pred,
            )
        )
    except Exception:
        results_list.append(_get_empty_metrics_ext("Support Vector Regression"))

    # Create dataframe and sort by Test R² (highest first)
    df = pd.DataFrame(results_list)
    df = df.sort_values("Test R²", ascending=False).reset_index(drop=True)
    return df


def predict_earthquakes_by_country(data, future_months=12, min_history_months=18):
    """
    Predict the number of earthquakes for each country over the next N months
    using a simple per-country Exponential Smoothing model.

    Parameters
    ----------
    data : pd.DataFrame
        Full earthquake dataset with 'time' and 'country' columns.
    future_months : int
        Number of months to forecast ahead (default 12).
    min_history_months : int
        Minimum months of data a country must have to be included (default 18).

    Returns
    -------
    pd.DataFrame
        Columns: country, historical_avg, predicted_total, predicted_monthly,
                 trend (Up / Stable / Down), confidence (High / Medium / Low)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

    df = data.copy()
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.tz_localize(None)
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

    results = []
    for country, grp in df.groupby("country"):
        monthly = grp.groupby("month").size().sort_index()
        if len(monthly) < min_history_months:
            continue

        y = monthly.values.astype(float)
        hist_avg = float(np.mean(y))

        # Try seasonal ES, fall back to trend-only, fall back to mean
        forecast_vals = None
        confidence = "Low"
        try:
            if len(y) >= 24:
                model = ES(
                    y,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated",
                ).fit(optimized=True, disp=False)
                confidence = "High"
            else:
                model = ES(
                    y, trend="add", seasonal=None, initialization_method="estimated"
                ).fit(optimized=True, disp=False)
                confidence = "Medium"
            forecast_vals = model.forecast(steps=future_months)
        except Exception:
            forecast_vals = np.full(future_months, hist_avg)

        forecast_vals = np.maximum(forecast_vals, 0)
        predicted_total = float(np.sum(forecast_vals))
        predicted_monthly = float(np.mean(forecast_vals))

        # Trend direction: compare last 3 months mean vs forecast mean
        recent_avg = float(np.mean(y[-3:])) if len(y) >= 3 else hist_avg
        pct_change = (predicted_monthly - recent_avg) / max(recent_avg, 1) * 100
        if pct_change > 5:
            trend = "↑ Up"
        elif pct_change < -5:
            trend = "↓ Down"
        else:
            trend = "→ Stable"

        results.append(
            {
                "country": country,
                "historical_avg": round(hist_avg, 1),
                "predicted_total": round(predicted_total, 0),
                "predicted_monthly": round(predicted_monthly, 1),
                "trend": trend,
                "confidence": confidence,
                "pct_change": round(pct_change, 1),
            }
        )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(
            "predicted_total", ascending=False
        ).reset_index(drop=True)
    return result_df


def _country_name_to_iso3(name):
    """
    Convert a country name string to its ISO 3166-1 alpha-3 code.
    Returns None if the name cannot be matched, so those rows are
    silently excluded from the choropleth (they'd show as blank anyway).
    """
    try:
        import pycountry

        results = pycountry.countries.search_fuzzy(str(name))
        if results:
            return results[0].alpha_3
    except Exception:
        pass
    return None


def plot_country_prediction_heatmap(pred_df, raw_data=None):
    """
    Create a bubble map of predicted earthquake counts by country.

    Uses px.scatter_map (OpenStreetMap tiles) so it renders reliably in
    Plotly 6 without any CDN dependency. Country centroids are derived from
    the raw earthquake data when provided, otherwise from a built-in lookup.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Output from predict_earthquakes_by_country().
    raw_data : pd.DataFrame or None
        Original earthquake DataFrame with latitude/longitude columns.
        Used to compute per-country centroids for bubble placement.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if pred_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No country prediction data available.")
        return fig

    df = pred_df.copy()

    # Compute country centroids from raw earthquake data
    if raw_data is not None and not raw_data.empty:
        centroids = (
            raw_data.dropna(subset=["latitude", "longitude", "country"])
            .groupby("country")[["latitude", "longitude"]]
            .mean()
            .reset_index()
        )
        df = df.merge(centroids, on="country", how="left")
    else:
        # Fallback: rough centroids for common seismic countries
        _fallback = {
            "Indonesia": (-2.5, 118.0),
            "Japan": (36.2, 138.3),
            "Chile": (-30.0, -71.0),
            "Philippines": (12.9, 121.8),
            "Turkey": (39.0, 35.0),
            "United States": (38.0, -97.0),
            "Mexico": (23.6, -102.5),
            "Iran": (32.4, 53.7),
            "Papua New Guinea": (-6.3, 143.9),
            "New Zealand": (-40.9, 174.9),
            "Peru": (-9.2, -75.0),
            "Russia": (61.5, 105.3),
            "India": (20.6, 79.0),
            "China": (35.9, 104.2),
            "Greece": (39.1, 21.8),
            "Italy": (41.9, 12.6),
        }
        df["latitude"] = df["country"].map(lambda c: _fallback.get(c, (None, None))[0])
        df["longitude"] = df["country"].map(lambda c: _fallback.get(c, (None, None))[1])

    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Could not compute country centroids for map.")
        return fig

    # Normalise bubble size: clamp to [6, 60] pixel range
    max_val = df["predicted_total"].max()
    df["bubble_size"] = (df["predicted_total"] / max_val * 54 + 6).round(1)

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        size="bubble_size",
        color="predicted_total",
        hover_name="country",
        hover_data={
            "bubble_size": False,
            "latitude": False,
            "longitude": False,
            "predicted_total": True,
            "predicted_monthly": True,
            "historical_avg": True,
            "trend": True,
            "confidence": True,
            "pct_change": True,
        },
        color_continuous_scale=[
            [0.0, "#1e3a5f"],
            [0.25, "#0369a1"],
            [0.5, "#0ea5a4"],
            [0.75, "#f97316"],
            [1.0, "#dc2626"],
        ],
        labels={
            "predicted_total": "Predicted Total",
            "predicted_monthly": "Monthly Avg",
            "historical_avg": "Historical Avg",
            "trend": "Trend",
            "confidence": "Confidence",
            "pct_change": "% Change",
        },
        title=f"ML Predicted Earthquake Count by Country — {int(df['predicted_total'].sum()):,} total",
        zoom=1,
        height=580,
        map_style="open-street-map",
        size_max=60,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Predicted<br>Earthquakes"),
            thickness=14,
            len=0.7,
        ),
    )
    return fig
