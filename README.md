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
- matplotlib (≥3.4.0)
- seaborn (≥0.11.0)
- folium (≥0.12.0)
- branca (≥0.4.2)
- streamlit (≥1.0.0)
- streamlit-folium (≥0.4.0)
- plotly (≥5.0.0)
- reverse_geocoder (≥1.5.1)
- pycountry (≥22.3.5)
- scikit-learn (≥1.0.0) - for ML predictions

## Folder structure

```text
Earthquake Data Analyzer/
├── data/
│   ├── historical.csv
│   ├── historical_processed.csv
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
├── plot.py
├── visualization.py
└── README.md
```

## 🎯 Dashboard Features

### 📊 Charts Tab

- Magnitude distribution and statistics
- Depth vs. magnitude correlation analysis
- Earthquake trends over time
- Top countries by frequency (bar and pie charts)
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

- **Machine Learning Model**: Random Forest Regressor for earthquake frequency prediction
- **Data Source**: Last 3 years of complete historical earthquake data
- **Monthly Aggregation**: Earthquake counts grouped by month for stable trends
- **Performance Metrics**: Train/Test R² scores and RMSE
- **Visualizations**: Actual vs Predicted trends, feature importance, and 12-month forecast
- **Data Transparency**: Shows total samples, training/testing split details
- **Beginner-friendly Explanations**: Clear documentation of what the model does and limitations

## 🤖 Machine Learning: Earthquake Frequency Trend Forecasting

### What It Does

The ML module uses a **Random Forest Regressor** to forecast earthquake **frequency trends** over time using the last **3 years of global historical earthquake data**. It helps identify whether earthquake activity is increasing or decreasing globally.

### Why Random Forest?

Random Forest is chosen because:

- **Handles Non-linear Patterns**: Earthquake frequency doesn't follow a straight line
- **Works with Noisy Data**: Earthquake patterns are chaotic and irregular
- **No Complex Preprocessing**: Unlike polynomial regression, Random Forest doesn't require scaling or smoothing
- **Interpretable**: Easy to understand which features are important
- **Robust**: Averages multiple decision trees to reduce overfitting
- **Beginner-Friendly**: Simple to explain to students and non-technical users

### How It Works (Simplified)

1. **Data Selection**: Load last 3 years from complete historical earthquake dataset
2. **Data Grouping**: Count earthquakes by month for stability
3. **Date Conversion**: Convert dates to numeric values (days since earliest date)
4. **Train/Test Split**: 80% for training, 20% for testing (chronological order preserved)
5. **Model Training**: Build 100 decision trees, each learning different patterns
6. **Averaging**: Combine predictions from all trees for final result
7. **Forecast**: Extend trends 12 months into the future

### Model Details

- **Algorithm**: Random Forest Regressor
- **Number of Trees**: 100 (balanced for accuracy and speed)
- **Tree Depth**: Max 10 levels (prevents overfitting)
- **Features**: Time (converted to days since earliest date)
- **Target**: Monthly earthquake frequency (count)
- **Training Data**: 80% of monthly samples
- **Testing Data**: 20% of monthly samples

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
- **Small sample**: Only 36 months of data may not capture all patterns
- **No external factors**: Tectonic shifts, instrumentation changes not modeled
- **Historical bias**: Past earthquakes may not reflect future patterns

### Displayed Metrics

For the trained model, the dashboard shows:

| Metric                 | Explanation                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Total Samples**      | Number of months used (usually ~36)                          |
| **Training Samples**   | 80% of total (used to train the model)                       |
| **Testing Samples**    | 20% of total (used to evaluate model)                        |
| **Train R² Score**     | How well the model fits training data (0-1, higher = better) |
| **Test R² Score**      | How well the model predicts new data (0-1, higher = better)  |
| **Train RMSE**         | Average error on training data                               |
| **Test RMSE**          | Average prediction error on test data (lower = better)       |
| **Feature Importance** | How important "time" is for predictions (0-1)                |

### Visualizations

1. **Actual vs Predicted Trends**: Shows how well the model follows real earthquake patterns
2. **Prediction Accuracy Scatter Plot**: Points near diagonal line = accurate predictions
3. **Feature Importance Chart**: Shows the "Time" feature importance in the model
4. **12-Month Forecast**: Raw data (dots), model trend (line), and future forecast (dotted line)

### Usage

1. Navigate to the **🤖 ML Trend Forecasting** tab
2. The model automatically trains on the last 3 years of global historical data
3. Review all performance metrics
4. Examine the trend analysis and prediction graphs
5. Check the forecast for the next 12 months
6. Read the explanations and limitations honestly stated

## Data Strategy: Why Last 3 Years?

The model uses the **last 3 years** of complete historical data for these reasons:

| Timeframe          | Monthly Points | Pros                                                           | Cons                                                 |
| ------------------ | -------------- | -------------------------------------------------------------- | ---------------------------------------------------- |
| **3 Years (Used)** | ~36 months     | ✅ Recent patterns, good sample size, avoids outdated behavior | Recent trends only                                   |
| 5 Years            | ~60 months     | ✅ More data points                                            | ❌ Includes older patterns that may be less relevant |
| 1 Year             | ~12 months     | ✅ Very recent                                                 | ❌ Too few points, high volatility, poor trends      |

**Best Practice**: 3 years balances data volume with relevance, capturing current seismic behavior without stale historical noise.

## 📝 Files Description

- **main.py**: Main Streamlit application with dashboard interface and ML section
- **ml_prediction.py**: ML module with Random Forest model for earthquake frequency prediction
- **visualization.py**: Functions for creating charts and maps
- **load_data.py**: Functions for fetching and processing earthquake data
- **plot.py**: Additional plotting utilities
- **data/**: Earthquake data files by year
