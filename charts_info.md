# Dashboard Charts and Libraries Info

This file documents the dashboard charts and the current implementation details so the documentation stays in sync with the code. Key implementation notes (default values and thresholds) are included so presenters and reviewers see the same behaviour as the app.

Here is a breakdown of all the charts and visualizations used in the Earthquake Data Analyzer dashboard, organized by tab, along with the primary Python libraries used to generate them and current implementation details:

## 📊 Tab 1: Charts

- **Magnitude Distribution**: Plotly Express (`px.histogram`)
- **Depth vs Magnitude**: Plotly Express (`px.scatter`)
- **Earthquake Trend Over Time**: Plotly Express (`px.line`)
- **Correlation Heatmap**: Plotly Express (`px.imshow`)
- **Top Countries Bar Chart**: Plotly Express (`px.bar`)

## 🗺️ Tab 2: Maps

_(All maps are rendered in the dashboard using the `streamlit-folium` library)_

- **Earthquake Heatmap**: Folium (`folium.plugins.HeatMap`)
- **Magnitude-based Map**: Folium (`folium.CircleMarker` with Branca colormap)
- **Depth-based Map**: Folium (`folium.CircleMarker` with Branca colormap)
- **Interactive Marker Cluster Map**: Folium (`folium.plugins.MarkerCluster`)
- **Country Overview Map**: Folium (Combines `HeatMap` and `CircleMarker`)

## 🎬 Tab 3: Advanced

- **Magnitude Categories**: Streamlit Native (`st.bar_chart`)
- **Depth Distribution**: Streamlit Native (`st.bar_chart`)
- **Time Series Analysis**: Streamlit Native (`st.line_chart`)

Notes (Advanced tab):

- **Depth bins:** current implementation uses boundaries [0, 10, 30, 70, 700] km producing labels: "Shallow (0-10km)", "Moderate (10-30km)", "Deep (30-70km)", "Very Deep (70+km)". This matches `main.py`'s `depth_categories` logic.

## 🔗 Tab 4: 3D & Animation

- **Animated Timeline**: Plotly Express (`px.scatter_geo`)
- **3D Earthquake Visualization**: Plotly Express (`px.scatter_3d`)

## 🤖 Tab 5: ML Prediction

- **Actual vs Predicted Earthquake Frequency**: Plotly Graph Objects (`make_subplots`)
- **Error Analysis (Residuals)**: Plotly Graph Objects (`make_subplots`)
- **Categorized Confusion Matrix**: Plotly Express (`px.imshow`)
- **12-Month Trend Forecast**: Plotly Graph Objects (`go.Figure` and `go.Scatter`)

Notes (ML Prediction tab):

- **Hybrid Model:** The dashboard uses a Hybrid approach: Exponential Smoothing as a base trend model plus a residual learner (Gradient Boosting) to model remaining structure. This is implemented in `ml_prediction.train_ml_model()`.
- **Default history:** The UI defaults to using up to the last **5 years** (~60 months) of data for training (configurable via the slider).
- **Aggregation:** Monthly (`'M'`) is the default aggregation for stable trends; weekly (`'W'`) is available for finer granularity.
- **Categorization (Low/Medium/High):** Continuous counts are converted to categories using the **30th and 70th percentiles** of the training series (i.e. 30% / 70% thresholds). This reduces boundary instability on small test windows and matches the `plot_confusion_matrix()` / `get_classification_report_df()` logic.
- **Accuracy (%) definition:** Shown "Accuracy (%)" is derived from MAPE as `max(0.0, 100.0 - Test_MAPE)`. MAPE is computed as mean(abs((y_true - y_pred) / max(y_true, 1))) \* 100 to avoid division-by-zero on very small counts.

Visualizations in this tab:

- Actual vs Predicted: scatter and perfect-prediction reference line
- Residuals: distribution of errors and error scatter
- Categorized confusion matrix & classification report: categorical stats using 30/70 train percentiles
- Future forecast: 12-month (monthly) or 26-week (weekly) forecast drawn as a dotted line starting from the last shown historical point

## ℹ️ Tab 6: About

- _No charts are present in this tab (Text-only information, markdown, and expanders)._

--

If you update related code (model, thresholds, or chart bins), update this file and the `README.md` to keep documentation consistent. Related files to check: `main.py`, `ml_prediction.py`, and `test_model.py`.
