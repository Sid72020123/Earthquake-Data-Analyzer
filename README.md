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

- pandas
- matplotlib
- seaborn
- folium
- streamlit
- streamlit-folium
- plotly
- reverse_geocoder
- pycountry
- scikit-learn (for ML predictions)

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

### 🤖 ML Prediction Tab

- **Data Source**: Uses last **3 years of complete historical data** (independent of dashboard filters)
- **Linear Regression Model**: Predicts earthquake frequency trends
- **Weekly Aggregation**: Groups earthquakes by week for trend analysis
- **Training Visualization**: Shows how well the model fits the training data
- **Actual vs Predicted**: Scatter plots and time series comparing predictions with real earthquake counts
- **Future Forecast**: Extrapolates trends for the next 12 weeks
- **Model Metrics**: Displays R² score, MSE, and RMSE for model evaluation
- **Beginner-friendly Explanation**: Learn how the ML model works with clear documentation

## 🤖 Machine Learning: Earthquake Frequency Prediction

### What It Does

The ML module uses **Linear Regression** to predict earthquake **frequency trends** over time using the last **3 years of global historical earthquake data**. It helps identify whether earthquake activity is increasing or decreasing globally.

### Data Strategy: Why Last 3 Years?

The model uses the **last 3 years** of complete historical data (independent of dashboard filters) for these reasons:

| Timeframe                 | Weekly Points | Pros                                                                   | Cons                                                                |
| ------------------------- | ------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **3 Years (Recommended)** | ~156 weeks    | ✅ Recent seismic patterns, good sample size, avoids outdated behavior | Shorter term trends                                                 |
| 5 Years                   | ~260 weeks    | ✅ More data points                                                    | ❌ Includes very old patterns that may not reflect current activity |
| 1 Year                    | ~52 weeks     | ✅ Very recent                                                         | ❌ Too few data points, high volatility, poor trends                |

**Best Practice**: 3 years balances data volume with relevance, capturing current earthquake behavior without stale historical noise.

### How It Works

1. **Data Selection**: Loads last 3 years from complete historical dataset (not dashboard filters)
2. **Data Grouping**: Earthquake data is grouped by week and counted
3. **Timezone Handling**: Converts timezone-aware dates to naive for clean period grouping
4. **Date Conversion**: Dates are converted to numeric values (days since earliest date) for ML
5. **Train/Test Split**: 80% of data for training, 20% for testing
6. **Feature Scaling**: StandardScaler normalizes numeric date features
7. **Model Training**: Linear Regression learns the relationship between time and earthquake frequency
8. **Prediction**: The model generates predictions and extrapolates future trends for 12 weeks

### Model Details

- **Algorithm**: Linear Regression (simple, interpretable, beginner-friendly)
- **Features**: Time (converted to days since the earliest date)
- **Target**: Weekly earthquake frequency (count)
- **Scaling**: StandardScaler for improved model performance
- **Evaluation**: R² score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE)

### What It DOES NOT Do

- Predict exact earthquake locations
- Predict exact earthquake times
- Predict specific earthquake magnitudes
- Account for external factors (tectonic changes, climate, etc.)
- Provide 100% accurate predictions (earthquake patterns are chaotic)

### Limitations

- Real earthquake patterns are complex and influenced by many factors
- The model only captures general trends, not short-term variations
- Results depend heavily on data quality and timeframe
- Historical earthquakes are underreported in many regions
- Small sample sizes may produce unreliable trends

### Usage

1. Navigate to the **🤖 ML Prediction** tab
2. The model automatically trains on the last 3 years of global historical data (dashboard filters don't affect ML)
3. Review the model performance metrics (R², MSE, RMSE)
4. Examine the training visualization, prediction graphs, and trend analysis
5. Check the info box explaining why 3 years was chosen
6. Read the model explanation to understand the results

## 📝 Files Description

- **main.py**: Main Streamlit application with dashboard interface
- **ml_prediction.py**: ML module for earthquake frequency prediction
- **visualization.py**: Functions for creating charts and maps
- **load_data.py**: Functions for fetching and processing earthquake data
- **plot.py**: Additional plotting utilities
- **data/**: Earthquake data files by year
