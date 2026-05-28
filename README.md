# Earthquake Data Analyzer

This is a simple and beginner-friendly Streamlit dashboard for exploring earthquake data. It lets you filter earthquakes by country, magnitude, and date, then view charts, maps, and summary statistics.

## How to run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Start the dashboard:

```bash
streamlit run main.py
```

## Required libraries

- pandas (≥1.3.0)
- folium (≥0.12.0)
- branca (≥0.4.2)
- streamlit (≥1.0.0)
- streamlit-folium (≥0.4.0)
- plotly (≥5.0.0)
- reverse_geocoder (≥1.5.1)
- pycountry (≥22.3.5)
- scikit-learn (≥1.0.0) - for ML predictions
- statsmodels (≥0.13.0) - for Exponential Smoothing and ARIMA models

## Folder structure

```text
Earthquake Data Analyzer/
├── data/
│   ├── historical.csv
│   ├── historical_processed.csv
│   ├── year_2012.csv
│   ├── year_2013.csv
│   ├── year_2014.csv
│   ├── year_2015.csv
│   ├── year_2016.csv
│   ├── year_2017.csv
│   ├── year_2018.csv
│   ├── year_2019.csv
│   ├── year_2020.csv
│   ├── year_2021.csv
│   ├── year_2022.csv
│   ├── year_2023.csv
│   ├── year_2024.csv
│   ├── year_2025.csv
│   └── year_2026.csv
├── load_data.py
├── main.py
├── ml_prediction.py
├── visualization.py
├── plot.py
└── README.md
```

## 🎯 Dashboard Features

### 📊 Charts Tab

- Magnitude distribution and statistics
- Depth vs. magnitude correlation analysis
- Earthquake trends over time
- Top countries by frequency (bar chart)
- Correlation heatmap of earthquake attributes

### 🗺️ Maps Tab

- **Heatmap**: Shows earthquake density across regions
- **Magnitude-based Map**: Circle size and color represent earthquake strength
- **Depth-based Map**: Color gradient from shallow to deep earthquakes
- **Marker Clusters**: Interactive clusters that expand on zoom
- **Country Overview**: Regional earthquake density patterns

### 🎬 Advanced Tab

- Magnitude and depth distribution categories
- Time series analysis of daily earthquake counts
- Categorical breakdowns for detailed insights

### 🔗 3D & Animation Tab

- **Animated Timeline**: Month-by-month animation showing earthquake patterns over time
- **3D Visualization**: Interactive 3D plot showing longitude, latitude, depth, and magnitude relationships

### 🤖 ML Trend Forecasting Tab

- **Machine Learning Model**: Hybrid Model (Exponential Smoothing + Gradient Boosting) for earthquake frequency trend prediction
- **Data Source**: Last 5 years of complete historical earthquake data (~60 months)
- **Monthly Aggregation**: Earthquake counts grouped by month for stable trends
- **Performance Metrics**: Train/Test R² scores and RMSE
- **Visualizations**: Actual vs Predicted trends, model fit, and 12-month forecast
- **Data Transparency**: Shows total samples, training/testing split details
- **Beginner-friendly Explanations**: Clear documentation of what the model does and limitations

## 🤖 Machine Learning: Earthquake Frequency Trend Forecasting

### What It Does

The ML module uses a **Hybrid Model (Exponential Smoothing + Gradient Boosting)** to forecast earthquake **frequency trends** over time using up to **5 years of global historical earthquake data** (configurable). It helps identify whether earthquake activity is increasing or decreasing globally.

### Why a Hybrid Model?

A Hybrid approach is chosen because:

- **Handles Noise**: Earthquake frequency is highly chaotic and noisy
- **Captures Trend**: Exponential Smoothing identifies the underlying baseline trend
- **Learns Patterns**: Gradient Boosting predicts the residuals (errors) of the base model, finding hidden correlations or seasonality
- **Better Accuracy**: Combining them typically outperforms a single model

### How It Works (Simplified)

1. **Data Selection**: Load up to 5 years of complete historical earthquake data (configurable via slider)
2. **Data Grouping**: Count earthquakes by month (or week) for stability
3. **Train/Test Split**: 80% for training, 20% for testing
4. **Model Training**: Train Exponential Smoothing on the data, then train Gradient Boosting on the residuals
5. **Forecast**: Extend trends 12 months (or 26 weeks) into the future using the combined model

### Model Details

- **Algorithm**: Hybrid (Exponential Smoothing + Gradient Boosting)
- **Features**: Monthly earthquake frequency counts, time index, month, quarter
- **Scaling**: None needed (uses raw monthly counts)
- **Training Data**: 80% of available time periods
- **Testing Data**: 20% of available time periods
- **Forecast Horizon**: 12 months (or 26 weeks) into the future

### What It DOES NOT Do

- Predict exact earthquake locations
- Predict exact earthquake times or dates
- Predict specific earthquake magnitudes
- Claim 100% accuracy
- Account for external factors (tectonic changes, climate, instrumentation)

### Important Limitations

- **Earthquake patterns are chaotic**: Real earthquakes are largely random and unpredictable
- **General trends only**: Model captures broad patterns, not short-term fluctuations
- **Data quality varies**: Some regions have better historical records than others
- **Small sample**: Only 60 months of data may not capture all patterns
- **No external factors**: Tectonic shifts, instrumentation changes not modeled
- **Historical bias**: Past earthquakes may not reflect future patterns

### Displayed Metrics

For the trained model, the dashboard shows:

| Metric               | Explanation                                                  |
| -------------------- | ------------------------------------------------------------ |
| **Total Samples**    | Number of time periods used (months or weeks)                |
| **Training Samples** | 80% of total (used to train the model)                       |
| **Testing Samples**  | 20% of total (used to evaluate model)                        |
| **Train R² Score**   | How well the model fits training data (0-1, higher = better) |
| **Test R² Score**    | How well the model predicts new data (0-1, higher = better)  |
| **Train RMSE**       | Average error on training data                               |
| **Test RMSE**        | Average prediction error on test data (lower = better)       |
| **Test MAPE**        | Mean absolute percentage error on test data                  |
| **Max Error**        | Largest single prediction error in the test period           |

### Visualizations

1. **Actual vs Predicted Trends**: Shows how well the model follows real earthquake patterns
2. **Prediction Accuracy Scatter Plot**: Points near diagonal line = accurate predictions
3. **Categorized Confusion Matrix**: Low/Medium/High activity classification accuracy
4. **12-Month Forecast**: Raw data (dots), model trend (teal line), and future forecast (pink dotted line)

### Usage

1. Navigate to the **🤖 ML Trend Forecasting** tab
2. Adjust the history slider (1 to all available years) and aggregation period (Monthly/Weekly)
3. The model automatically trains on the selected global historical data
4. Review all performance metrics
5. Examine the trend analysis and prediction graphs
6. Check the forecast for the next 12 months (or 26 weeks)
7. Read the explanations and limitations honestly stated

## Data Strategy: Why Last 5 Years?

The model uses up to **5 years** of complete historical data by default, configurable via the dashboard slider:

| Timeframe             | Monthly Points | Pros                                        | Cons                                                |
| --------------------- | -------------- | ------------------------------------------- | --------------------------------------------------- |
| **5 Years (Default)** | ~60 months     | ✅ More data points, better trend stability | May include older patterns that differ from present |
| 5 Years (Used)        | ~60 months     | ✅ Recent patterns, good sample size        | Less data for model training                        |
| 1 Year                | ~12 months     | ✅ Very recent                              | ❌ Too few points, high volatility, poor trends     |

**Best Practice**: More history generally yields more stable trend estimates. Use the slider in the dashboard to experiment.

## 📝 Files Description

- **main.py**: Main Streamlit application with dashboard interface and ML section
- **ml_prediction.py**: ML module with Hybrid Model (Exponential Smoothing + Gradient Boosting) for earthquake frequency trend prediction
- **visualization.py**: Functions for creating charts and maps
- **load_data.py**: Functions for fetching and processing earthquake data
- **plot.py**: Standalone reference script for a simple marker cluster map (kept for reference; functionality is in visualization.py)
- **data/**: Earthquake data files (historical.csv, historical_processed.csv, and per-year CSVs from 2012 to 2026)
