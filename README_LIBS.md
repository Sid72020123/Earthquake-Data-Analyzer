# Libraries & Functions Used

This document lists the Python libraries used by the Earthquake Data Analyzer project and the primary functions, classes, or attributes from each library that the codebase calls.

---

## pandas

- `pd.read_csv`
- `pd.to_datetime`
- `pd.DataFrame` methods: `.groupby()`, `.size()`, `.value_counts()`, `.dropna()`, `.reset_index()`, `.merge()`, `.head()`, `.sample()`, `.concat()`, `.astype()`, `.copy()`, `.to_csv()`
- Datetime accessors: `.dt.to_period()`, `.dt.to_timestamp()`, `.dt.tz_convert()`, `.dt.normalize()`, `.dt.strftime()`

Short explanations (how used in this project):

- `pd.read_csv`: load CSV datasets (yearly and processed histories).
- `pd.to_datetime`: parse and normalize timestamp columns for grouping and filtering.
- `.groupby()/.size()`: aggregate counts by period (month/week) for trend charts and ML input.
- `.value_counts()`: compute top countries and categorical summaries.
- `.dropna()/.reset_index()`: clean and prepare dataframes for plotting/export.
- `.to_csv()`: persist intermediate and processed CSV files under `data/`.

## numpy

- Array creation and ops: `np.asarray`, `np.array`, `np.concatenate`, `np.hstack`
- Numeric utilities: `np.mean`, `np.max`, `np.min`, `np.percentile`, `np.abs`, `np.maximum`

Short explanations:

- `np.percentile`: used to compute 30th/70th percentiles for categorizing Low/Medium/High activity.
- `np.maximum`: clamp negative forecasts to zero and avoid division-by-zero issues.

## plotly (plotly.express / plotly.graph_objects / plotly.subplots)

- `plotly.express` (aliased `px`): `px.histogram`, `px.scatter`, `px.imshow`, `px.bar`, `px.line`, `px.scatter_geo`, `px.scatter_map`, `px.scatter_3d`, `px.scatter_map` (animation)
- `plotly.graph_objects` (aliased `go`): `go.Figure`, `go.Scatter`
- `plotly.subplots.make_subplots`
- Figure methods: `.add_trace()`, `.update_layout()`, `.update_xaxes()`, `.update_yaxes()`, `.update_traces()`

Short explanations:

- `px.histogram`, `px.scatter`, `px.line`, `px.bar`: quick high-level chart creation for magnitude, depth, trends, and top-country bars.
- `px.scatter_map`, `px.scatter_geo`, `px.scatter_3d`: animated timeline and 3D visualizations for geographic/time animations.
- `go.Figure` / `go.Scatter`: custom multi-trace figures used in ML Actual vs Predicted and forecast plots.

## streamlit

- Page & layout: `st.set_page_config`, `st.columns`, `st.tabs`, `st.expander`, `st.container`
- Widgets: `st.selectbox`, `st.slider`, `st.radio`, `st.checkbox`, `st.date_input`, `st.metric`, `st.table`, `st.dataframe`
- Display helpers: `st.plotly_chart`, `st.markdown`, `st.caption`, `st.info`, `st.warning`, `st.error`, `st.spinner`, `st.iframe`, `st.divider`
- Theme/options: `st.get_option`

Short explanations:

- `st.set_page_config`, `st.columns`, `st.tabs`, `st.expander`: layout and structure of the dashboard pages.
- Widgets (`st.selectbox`, `st.slider`, `st.radio`, `st.checkbox`, `st.date_input`): user controls for filters and ML parameters.
- Displays (`st.plotly_chart`, `st.dataframe`, `st.metric`, `st.info`, `st.warning`, `st.spinner`): present charts, tables and status feedback.

## streamlit_folium

- `st_folium` (embed Folium maps in Streamlit)

Short explanation:

- `st_folium`: renders interactive Folium maps inside Streamlit layout while preserving map interactivity.

## folium

- Map & layers: `folium.Map`, `folium.CircleMarker`, `folium.Popup`
- Map saving: `.save()`
- `folium.plugins`: `HeatMap`, `MarkerCluster`

Short explanations:

- `folium.Map`, `folium.CircleMarker`, `folium.Popup`: build interactive maps with per-quake markers and popups.
- `HeatMap`: render density heatmaps of earthquake locations.
- `MarkerCluster`: group dense markers for interactive zooming.

## branca

- `branca.colormap.LinearColormap`

Short explanation:

- `LinearColormap`: color scales for magnitude/depth choropleths and map legends.

## scikit-learn (sklearn)

- Models & utilities:
    - `sklearn.ensemble.GradientBoostingRegressor`
    - `sklearn.ensemble.RandomForestRegressor`
    - `sklearn.linear_model.Ridge`
    - `sklearn.svm.SVR`
- Metrics: `sklearn.metrics.r2_score`, `sklearn.metrics.mean_squared_error`, `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.confusion_matrix`, `sklearn.metrics.classification_report`

Short explanations:

- `GradientBoostingRegressor`: residual learner in the Hybrid model to predict errors from Exponential Smoothing.
- `RandomForestRegressor`, `Ridge`, `SVR`: alternate models evaluated in `compare_models()`.
- Metrics (`r2_score`, `mean_squared_error`, `mean_absolute_error`): used to evaluate and display model performance.
- `confusion_matrix` / `classification_report`: create the categorized (Low/Medium/High) evaluation table and confusion matrix.

## statsmodels

- Time series models:
    - `statsmodels.tsa.holtwinters.ExponentialSmoothing`
    - `statsmodels.tsa.arima.model.ARIMA`

Short explanations:

- `ExponentialSmoothing`: base time-series trend estimator used in the Hybrid model and per-country forecasts.
- `ARIMA`: evaluated as a comparison model in `compare_models()` (used only when appropriate data size).

## reverse_geocoder

- `reverse_geocoder.search` (lookup lat/lon → country code)

Short explanation:

- `reverse_geocoder.search`: reverse-geocodes latitude/longitude pairs to ISO country codes during dataset processing.

## pycountry

- Country lookup: `pycountry.countries.get`, `pycountry.countries.search_fuzzy`

Short explanation:

- Convert ISO country codes to full country names and handle fuzzy matches for common country name variations.

## os (builtin)

- Filesystem helpers: `os.path.exists`, `os.makedirs` / `os.makedirs(..., exist_ok=True)`

Short explanation:

- File/directory existence checks and creation when saving fetched yearly CSVs and processed outputs.

## warnings (builtin)

- `warnings.catch_warnings`, `warnings.filterwarnings`

Short explanation:

- Used to suppress known non-critical warnings from time-series fitting routines during model comparison.

---
