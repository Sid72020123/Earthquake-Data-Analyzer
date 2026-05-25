# Dashboard Charts and Libraries Info

Here is a breakdown of all the charts and visualizations used in the Earthquake Data Analyzer dashboard, organized by their respective tabs, along with the primary Python libraries used to generate them:

## 📊 Tab 1: Charts

- **Magnitude Distribution**: Seaborn & Matplotlib (`sns.histplot`)
- **Depth vs Magnitude**: Seaborn & Matplotlib (`sns.scatterplot`)
- **Earthquake Trend Over Time**: Seaborn & Matplotlib (`sns.lineplot`)
- **Correlation Heatmap**: Seaborn & Matplotlib (`sns.heatmap`)
- **Top Countries Bar Chart**: Seaborn & Matplotlib (`sns.barplot`)
- **Top Countries Pie Chart**: Matplotlib (`ax.pie`)

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

## 🔗 Tab 4: 3D & Animation

- **Animated Timeline**: Plotly Express (`px.scatter_geo`)
- **3D Earthquake Visualization**: Plotly Express (`px.scatter_3d`)

## 🤖 Tab 5: ML Prediction

- **Actual vs Predicted Earthquake Frequency**: Matplotlib (`plt.plot` and `plt.scatter`)
- **Error Analysis (Residuals)**: Seaborn & Matplotlib (`sns.histplot` and `plt.scatter`)
- **Categorized Confusion Matrix**: Seaborn & Matplotlib (`sns.heatmap`)
- **12-Month Trend Forecast**: Plotly Graph Objects (`go.Figure` and `go.Scatter`)

## ℹ️ Tab 6: About

- _No charts are present in this tab (Text-only information, markdown, and expanders)._
