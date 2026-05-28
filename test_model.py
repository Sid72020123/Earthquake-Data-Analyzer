import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from ml_prediction import (
    prepare_time_series_data,
    train_ml_model,
)

# Load data
data = pd.read_csv(
    "data/historical_processed.csv", parse_dates=["time"], low_memory=False
)
data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True)
data = data.dropna(subset=["time", "latitude", "longitude", "mag", "depth", "country"])

# Use full history (same as dashboard default) with partial-month trimming
frequency_data = prepare_time_series_data(data, period="M")
print(f"Using {len(frequency_data)} months of data "
      f"({frequency_data['time'].iloc[0].date()} → {frequency_data['time'].iloc[-1].date()})")

# Split data
split_index = int(len(frequency_data) * 0.8)
train_data = frequency_data.iloc[:split_index]
test_data = frequency_data.iloc[split_index:]
y_train = train_data["count"].values
y_test = test_data["count"].values

print(f"Train: {split_index} months, Test: {len(y_test)} months")
print(f"Test period: {test_data['time'].iloc[0].date()} → {test_data['time'].iloc[-1].date()}")

print("=" * 60)
print("📊 COMPARING MODELS FOR EARTHQUAKE FREQUENCY PREDICTION")
print("=" * 60)

# 1. Current Hybrid Model (ES + GBM)
print("\n1️⃣  HYBRID MODEL (ES + Gradient Boosting) (Current Model)")
print("-" * 60)
ml_results = train_ml_model(frequency_data, test_size=0.2)
print(f"Train R²: {ml_results['train_r2']:.4f}")
print(f"Test R²:  {ml_results['test_r2']:.4f}")
print(f"Train RMSE: {ml_results['train_rmse']:.4f}")
print(f"Test RMSE:  {ml_results['test_rmse']:.4f}")

# Initialize all model variables before try blocks
exp_test_r2 = 0.0
exp_train_r2 = 0.0
ridge_test_r2 = 0.0
ridge_train_r2 = 0.0

# 2. Exponential Smoothing Model
print("\n2️⃣  EXPONENTIAL SMOOTHING (Better for Time Series)")
print("-" * 60)
try:
    # Fit exponential smoothing on training data
    exp_smooth = ExponentialSmoothing(
        y_train, trend="add", seasonal=None, initialization_method="estimated"
    )
    exp_model = exp_smooth.fit(optimized=True)

    # Make predictions
    exp_train_pred = exp_model.fittedvalues
    exp_test_pred = exp_model.forecast(steps=len(y_test))

    # Calculate metrics
    exp_train_r2 = r2_score(y_train, exp_train_pred)
    exp_test_r2 = r2_score(y_test, exp_test_pred)
    exp_train_rmse = np.sqrt(mean_squared_error(y_train, exp_train_pred))
    exp_test_rmse = np.sqrt(mean_squared_error(y_test, exp_test_pred))

    print(f"Train R²: {exp_train_r2:.4f}")
    print(f"Test R²:  {exp_test_r2:.4f}")
    print(f"Train RMSE: {exp_train_rmse:.4f}")
    print(f"Test RMSE:  {exp_test_rmse:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 3. Moving Average Model
print("\n3️⃣  MOVING AVERAGE (Smoothing Noise)")
print("-" * 60)
window_size = 3
ma_train = np.convolve(y_train, np.ones(window_size) / window_size, mode="valid")
ma_train_actual = y_train[window_size - 1 :]  # Align sizes

# For test, use last value as baseline
ma_test_pred = np.full(len(y_test), np.mean(y_train))

ma_train_r2 = r2_score(ma_train_actual, ma_train)
ma_test_r2 = r2_score(y_test, ma_test_pred)
ma_train_rmse = np.sqrt(mean_squared_error(ma_train_actual, ma_train))
ma_test_rmse = np.sqrt(mean_squared_error(y_test, ma_test_pred))

print(f"Train R²: {ma_train_r2:.4f}")
print(f"Test R²:  {ma_test_r2:.4f}")
print(f"Train RMSE: {ma_train_rmse:.4f}")
print(f"Test RMSE:  {ma_test_rmse:.4f}")

# 4. Ridge Regression (Regularized Linear Regression)
print("\n4️⃣  RIDGE REGRESSION (Handles Noise Better)")
print("-" * 60)
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

    print(f"Train R²: {ridge_train_r2:.4f}")
    print(f"Test R²:  {ridge_test_r2:.4f}")
    print(f"Train RMSE: {ridge_train_rmse:.4f}")
    print(f"Test RMSE:  {ridge_test_rmse:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 5. ARIMA Model (AutoRegressive Integrated Moving Average)
print("\n5️⃣  ARIMA (Time Series Forecasting)")
print("-" * 60)
# Initialize variables in case of error
arima_test_r2 = 0.0
arima_train_r2 = 0.0
arima_test_rmse = 0.0
arima_train_rmse = 0.0

try:
    # Try auto ARIMA configuration (simple order)
    arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()

    arima_train_pred = arima_model.fittedvalues
    arima_test_pred = arima_model.get_forecast(steps=len(y_test)).predicted_mean

    arima_train_r2 = r2_score(y_train, arima_train_pred)
    arima_test_r2 = r2_score(y_test, arima_test_pred)
    arima_train_rmse = np.sqrt(mean_squared_error(y_train, arima_train_pred))
    arima_test_rmse = np.sqrt(mean_squared_error(y_test, arima_test_pred))

    print(f"Train R²: {arima_train_r2:.4f}")
    print(f"Test R²:  {arima_test_r2:.4f}")
    print(f"Train RMSE: {arima_train_rmse:.4f}")
    print(f"Test RMSE:  {arima_test_rmse:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 6. Seasonal Naive (Use value from 12 months ago)
print("\n6️⃣  SEASONAL NAIVE (Repeat from 12 Months Ago)")
print("-" * 60)
# Initialize variables in case of error
sn_test_r2 = 0.0
sn_train_r2 = 0.0
sn_test_rmse = 0.0
sn_train_rmse = 0.0

try:
    season = 12
    # For training, use lagged values
    if len(y_train) >= season:
        sn_train_actual = y_train[season:]
        sn_train_pred = y_train[:-season]  # lag by one season

        # For test, use corresponding value from training (or last available)
        sn_test_pred = np.full(
            len(y_test),
            y_train[-season] if len(y_train) >= season else np.mean(y_train),
        )

        sn_train_r2 = r2_score(sn_train_actual, sn_train_pred)
        sn_test_r2 = r2_score(y_test, sn_test_pred)
        sn_train_rmse = np.sqrt(mean_squared_error(sn_train_actual, sn_train_pred))
        sn_test_rmse = np.sqrt(mean_squared_error(y_test, sn_test_pred))

        print(f"Train R²: {sn_train_r2:.4f}")
        print(f"Test R²:  {sn_test_r2:.4f}")
        print(f"Train RMSE: {sn_train_rmse:.4f}")
        print(f"Test RMSE:  {sn_test_rmse:.4f}")
    else:
        print("Not enough data for seasonal naive (need >= 12 months)")
except Exception as e:
    print(f"Error: {e}")

# 7. Baseline: Mean Prediction
print("\n7️⃣  BASELINE (Predict Mean Value)")
print("-" * 60)
baseline_pred_train = np.full(len(y_train), np.mean(y_train))
baseline_pred_test = np.full(len(y_test), np.mean(y_train))

baseline_train_r2 = r2_score(y_train, baseline_pred_train)
baseline_test_r2 = r2_score(y_test, baseline_pred_test)
baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred_test))

print(f"Train R²: {baseline_train_r2:.4f}")
print(f"Test R²:  {baseline_test_r2:.4f}")
print(f"Train RMSE: {baseline_train_rmse:.4f}")
print(f"Test RMSE:  {baseline_test_rmse:.4f}")

print("\n" + "=" * 60)
print("🏆 COMPARISON SUMMARY:")
print("=" * 60)
models_summary = [
    ("Hybrid (ES + GBM)", ml_results["test_r2"]),
    ("Exponential Smoothing", exp_test_r2),
    ("Moving Average", ma_test_r2),
    ("Ridge Regression", ridge_test_r2),
    ("ARIMA", arima_test_r2),
    ("Seasonal Naive", sn_test_r2),
    ("Baseline (Mean)", baseline_test_r2),
]

# Sort by test R² (highest first)
models_summary.sort(key=lambda x: x[1], reverse=True)

for i, (model_name, r2) in enumerate(models_summary, 1):
    print(f"{i}. {model_name:.<30} Test R² = {r2:>8.4f}")

print("=" * 60)
best_model = models_summary[0]
print(f"✅ BEST MODEL: {best_model[0]} (Test R² = {best_model[1]:.4f})")
print("=" * 60)
