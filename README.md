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

- **Linear Regression Model**: Predicts earthquake frequency trends
- **Training Visualization**: Shows how well the model fits the data
- **Actual vs Predicted**: Compares predictions with real earthquake counts
- **Future Forecast**: Extrapolates trends for the next 12 weeks
- **Model Metrics**: Displays R² score, MSE, and RMSE
- **Beginner-friendly Explanation**: Learn how the ML model works without needing a data science background

## 🤖 Machine Learning: Earthquake Frequency Prediction

### What It Does

The ML module uses **Linear Regression** to predict earthquake **frequency trends** over time. It helps identify whether earthquake activity is increasing or decreasing in the selected region and time period.

### How It Works

1. **Data Grouping**: Earthquake data is grouped by week and counted
2. **Date Conversion**: Dates are converted to numeric values for the model
3. **Train/Test Split**: 80% of data for training, 20% for testing
4. **Model Training**: Linear Regression learns the relationship between time and earthquake frequency
5. **Prediction**: The model generates predictions and extrapolates future trends

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

1. Select your desired filters (country, date range, magnitude range)
2. The model automatically trains on the filtered data
3. Review the model performance metrics
4. Examine the prediction graphs and trend analysis
5. Read the model explanation to understand the results

## 📝 Files Description

- **main.py**: Main Streamlit application with dashboard interface
- **ml_prediction.py**: ML module for earthquake frequency prediction
- **visualization.py**: Functions for creating charts and maps
- **load_data.py**: Functions for fetching and processing earthquake data
- **plot.py**: Additional plotting utilities
- **data/**: Earthquake data files by year
