"""Streamlit dashboard for the Earthquake Data Analyzer project."""

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium

from visualization import (
    create_3d_earthquake_visualization,
    create_animated_timeline,
    create_folium_heatmap,
    create_marker_cluster_map,
    create_magnitude_based_map,
    create_depth_based_map,
    create_country_region_map,
    plot_correlation_heatmap,
    plot_depth_vs_magnitude,
    plot_earthquake_trend,
    plot_magnitude_distribution,
    plot_top_countries_bar_chart,
)

from ml_prediction import (
    prepare_time_series_data,
    train_ml_model,
    plot_actual_vs_predicted,
    create_prediction_plotly,
    plot_confusion_matrix,
    get_classification_report_df,
    get_model_explanation,
    get_ml_data_from_full_history,
    compare_models,
    predict_earthquakes_by_country,
    plot_country_prediction_heatmap,
)

DATA_PATH = "data/historical_processed.csv"


st.set_page_config(
    page_title="Earthquake Data Analyzer",
    page_icon="🌍",
    layout="wide",
)


def apply_page_style():
    """Add custom CSS for a professional, modern dashboard look."""

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            html, body, .stApp {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .stApp {
                background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 50%, #f0fff4 100%);
            }

            /* ── Title block ──────────────────────────────────────────────────── */
            .title-block {
                padding: 2rem 2rem;
                background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
                border-radius: 1.5rem;
                box-shadow: 0 4px 24px rgba(14,165,164,0.10), 0 1px 4px rgba(0,0,0,0.06);
                margin-bottom: 1.5rem;
                border-left: 6px solid #0ea5a4;
                border-top: 1px solid rgba(14,165,164,0.15);
            }
            .title-block h1 {
                margin-bottom: 0.35rem;
                color: #0f172a;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            .title-block .subtitle {
                color: #475569;
                font-size: 1.05rem;
                font-weight: 400;
                margin: 0;
            }
            .title-block .badge {
                display: inline-block;
                background: linear-gradient(90deg, #0ea5a4, #06b6d4);
                color: #fff;
                border-radius: 999px;
                padding: 2px 14px;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.5px;
                margin-right: 6px;
                vertical-align: middle;
            }

            /* ── Sidebar background ───────────────────────────────────────────── */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
            }

            /* All plain text & labels inside sidebar */
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stCaption {
                color: #cbd5e1 !important;
            }

            /* ── Selectbox / Dropdown ─────────────────────────────────────────── */
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="base-input"] {
                background-color: #1e3a5f !important;
                border-color: #334155 !important;
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="base-input"] * {
                color: #e2e8f0 !important;
            }
            /* The selected value text */
            [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stWidgetLabel"] + div span,
            [data-testid="stSidebar"] [data-baseweb="select"] span {
                color: #e2e8f0 !important;
            }

            /* ── Date input ───────────────────────────────────────────────────── */
            [data-testid="stSidebar"] [data-testid="stDateInputContainer"] > div,
            [data-testid="stSidebar"] input[type="text"],
            [data-testid="stSidebar"] input[type="date"],
            [data-testid="stSidebar"] input {
                background-color: #1e3a5f !important;
                border-color: #334155 !important;
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] input::placeholder {
                color: #64748b !important;
            }

            /* ── Multiselect tags ────────────────────────────────────────────── */
            [data-testid="stSidebar"] [data-baseweb="tag"] {
                background-color: #0ea5a4 !important;
                color: #ffffff !important;
            }

            /* ── Slider track & thumb ────────────────────────────────────────── */
            [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div {
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
                background-color: #0ea5a4 !important;
                border-color: #06b6d4 !important;
            }
            /* Slider value tooltip */
            [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="tooltip"] {
                background-color: #0ea5a4 !important;
                color: #fff !important;
            }

            /* ── Checkbox ────────────────────────────────────────────────────── */
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
                color: #cbd5e1 !important;
            }

            /* ── Divider ─────────────────────────────────────────────────────── */
            [data-testid="stSidebar"] hr {
                border-color: #334155 !important;
            }

            /* Dropdown option list that renders outside sidebar (portal) */
            [data-baseweb="popover"] [data-baseweb="menu"],
            [data-baseweb="popover"] ul {
                background-color: #1e293b !important;
                border-color: #334155 !important;
            }
            [data-baseweb="popover"] [role="option"],
            [data-baseweb="popover"] li {
                color: #e2e8f0 !important;
                background-color: #1e293b !important;
            }
            [data-baseweb="popover"] [role="option"]:hover,
            [data-baseweb="popover"] li:hover {
                background-color: #0ea5a4 !important;
                color: #ffffff !important;
            }

            /* ── Metric cards ────────────────────────────────────────────────── */
            [data-testid="stMetric"] {
                background: #ffffff;
                border-radius: 1rem;
                padding: 0.9rem 1.1rem;
                box-shadow: 0 2px 12px rgba(14,165,164,0.08);
                border-top: 3px solid #0ea5a4;
            }

            /* ── Tab styling ─────────────────────────────────────────────────── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
                background: rgba(14,165,164,0.06);
                border-radius: 12px;
                padding: 4px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                padding: 6px 18px;
                font-weight: 500;
            }

            /* ── Dark mode overrides ─────────────────────────────────────────── */
            [data-theme="dark"] .stApp {
                background: linear-gradient(135deg, #071027 0%, #0b1a2e 50%, #071020 100%);
            }
            [data-theme="dark"] .title-block {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                box-shadow: 0 4px 24px rgba(6,182,212,0.10);
                border-left: 6px solid #06b6d4;
            }
            [data-theme="dark"] .title-block h1 { color: #e2e8f0; }
            [data-theme="dark"] .title-block .subtitle { color: #94a3b8; }
            [data-theme="dark"] [data-testid="stMetric"] {
                background: #1e293b;
                border-top: 3px solid #06b6d4;
                box-shadow: 0 2px 12px rgba(6,182,212,0.10);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    """Load the processed earthquake dataset and clean the key columns."""

    try:
        data = pd.read_csv(DATA_PATH, parse_dates=["time"], low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return data

    data = data.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True)
    data = data.dropna(
        subset=["time", "latitude", "longitude", "mag", "depth", "country"]
    )
    data["country"] = data["country"].fillna("Unknown")
    return data


def filter_data(
    data,
    country_choice,
    magnitude_range,
    selected_dates,
    depth_range=None,
    region_choice="All Regions",
):
    """Filter the dataset using the sidebar controls."""

    filtered = data.copy()

    if country_choice != "All Countries":
        filtered = filtered[filtered["country"] == country_choice]

    filtered = filtered[
        (filtered["mag"] >= magnitude_range[0])
        & (filtered["mag"] <= magnitude_range[1])
    ]

    if depth_range:
        filtered = filtered[
            (filtered["depth"] >= depth_range[0])
            & (filtered["depth"] <= depth_range[1])
        ]

    if region_choice != "All Regions":
        filtered = filtered[filtered["region"] == region_choice]

    start_date, end_date = selected_dates
    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = (
        pd.Timestamp(end_date, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    filtered = filtered[
        (filtered["time"] >= start_timestamp) & (filtered["time"] <= end_timestamp)
    ]

    return filtered


def show_metric_cards(data):
    """Display the key statistics cards."""

    total_earthquakes = len(data)
    average_magnitude = data["mag"].mean()
    maximum_magnitude = data["mag"].max()
    minimum_magnitude = data["mag"].min()
    average_depth = data["depth"].mean() if "depth" in data.columns else 0
    max_depth = data["depth"].max() if "depth" in data.columns else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("🌍 Total Earthquakes", f"{total_earthquakes:,}")
    col2.metric("📊 Avg Magnitude", f"{average_magnitude:.2f}")
    col3.metric("📈 Max Magnitude", f"{maximum_magnitude:.2f}")
    col4.metric("📉 Min Magnitude", f"{minimum_magnitude:.2f}")
    col5.metric("🔻 Avg Depth (km)", f"{average_depth:.1f}")
    col6.metric("📍 Max Depth (km)", f"{max_depth:.1f}")


def render_charts_tab(filtered_data):
    """Renders the 'Charts' tab with various statistical plots."""
    st.subheader("Statistical Analysis")
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(plot_magnitude_distribution(filtered_data), width='stretch')
        st.plotly_chart(plot_depth_vs_magnitude(filtered_data), width='stretch')
    with right_col:
        st.plotly_chart(plot_correlation_heatmap(filtered_data), width='stretch')
        st.plotly_chart(plot_top_countries_bar_chart(filtered_data), width='stretch')

    # Full-width trend chart
    st.plotly_chart(plot_earthquake_trend(filtered_data), width='stretch')
    st.markdown("---")
    st.subheader("🚨 Top 10 Largest Earthquakes in View")
    top_10 = filtered_data.nlargest(10, "mag")[
        ["time", "place", "mag", "depth", "country"]
    ]
    top_10["time"] = (
        pd.to_datetime(top_10["time"]).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M")
    )
    top_10.columns = [
        "Time (UTC)",
        "Location",
        "Magnitude",
        "Depth (km)",
        "Country",
    ]
    st.dataframe(top_10, width='stretch')


def render_maps_tab(display_data, map_options):
    """Renders the 'Maps' tab with various Folium maps."""
    st.subheader("🔥 Earthquake Heatmap")
    st.caption(
        "Dense earthquake locations are highlighted. Zoom in to explore specific areas."
    )
    if map_options["show_heatmap"]:
        try:
            with st.spinner("Rendering heatmap..."):
                # Render folium map HTML and embed via st.iframe to avoid glitches
                m = create_folium_heatmap(display_data)
                map_html = m._repr_html_()
                st.iframe(map_html, height=610)
        except Exception as exc:
            st.error(f"Could not render the heatmap: {exc}")
    else:
        st.info("Enable heatmap in the sidebar filters to view this map.")

    col_map1, col_map2 = st.columns(2)

    with col_map1:
        st.subheader("📍 Magnitude-based Map")
        st.caption("Circle size and color represent earthquake magnitude.")
        if map_options["show_magnitude_map"]:
            try:
                with st.spinner("Rendering magnitude map..."):
                    st_folium(
                        create_magnitude_based_map(display_data),
                        width='stretch',
                        height=500,
                        returned_objects=[],
                    )
            except Exception as exc:
                st.error(f"Could not render the magnitude map: {exc}")
        else:
            st.info("Enable magnitude map in sidebar to view.")

    with col_map2:
        st.subheader("🌊 Depth-based Map")
        st.caption("Circle color represents depth (shallow to deep).")
        if map_options["show_depth_map"]:
            try:
                with st.spinner("Rendering depth map..."):
                    st_folium(
                        create_depth_based_map(display_data),
                        width='stretch',
                        height=500,
                        returned_objects=[],
                    )
            except Exception as exc:
                st.error(f"Could not render the depth map: {exc}")
        else:
            st.info("Enable depth map in sidebar to view.")

    st.subheader("🎯 Interactive Marker Cluster Map")
    st.caption(
        "Click clusters to zoom in. Individual markers appear at higher zoom levels."
    )
    if map_options["show_cluster_map"]:
        try:
            with st.spinner("Rendering cluster map..."):
                st_folium(
                    create_marker_cluster_map(display_data),
                    width='stretch',
                    height=600,
                    returned_objects=[],
                )
        except Exception as exc:
            st.error(f"Could not render the cluster map: {exc}")
    else:
        st.info("Enable marker clusters in the sidebar filters to view this map.")

    st.subheader("🌍 Country Overview Map")
    st.caption(
        "Heatmap showing earthquake density across regions with country markers."
    )
    if map_options["show_country_map"]:
        try:
            with st.spinner("Rendering country overview map..."):
                st_folium(
                    create_country_region_map(display_data),
                    width='stretch',
                    height=600,
                    returned_objects=[],
                )
        except Exception as exc:
            st.error(f"Could not render the country map: {exc}")
    else:
        st.info("Enable country overview map in sidebar to view.")


def render_advanced_tab(filtered_data):
    """Renders the 'Advanced' tab with categorical and time-series analysis."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Magnitude Categories")
        mag_categories = pd.cut(
            filtered_data["mag"],
            bins=[0, 4, 5, 6, 7, 10],
            labels=[
                "Minor (0-4)",
                "Light (4-5)",
                "Moderate (5-6)",
                "Strong (6-7)",
                "Major (7+)",
            ],
        )
        mag_dist = mag_categories.value_counts().sort_index()
        st.bar_chart(mag_dist)

    with col2:
        st.subheader("🔍 Depth Distribution")
        depth_categories = pd.cut(
            filtered_data["depth"],
            bins=[0, 10, 30, 70, 300],
            labels=[
                "Shallow (0-10km)",
                "Moderate (10-30km)",
                "Deep (30-70km)",
                "Very Deep (70+km)",
            ],
        )
        depth_dist = depth_categories.value_counts().sort_index()
        st.bar_chart(depth_dist)

    st.subheader("📈 Time Series Analysis")
    time_data = filtered_data.copy()
    # Keep as datetime so Streamlit treats the x-axis as a continuous time scale
    time_data["date"] = (
        pd.to_datetime(time_data["time"]).dt.tz_convert(None).dt.normalize()
    )
    daily_counts = time_data.groupby("date").size()
    st.line_chart(daily_counts)


def render_animation_tab(filtered_data, timeline_sample_size):
    """Renders the '3D & Animation' tab."""
    st.subheader("🎬 Animated Timeline")
    st.caption(
        "Animation progresses month by month. Watch earthquake patterns unfold over time!"
    )
    try:
        # Allow the timeline slider to control how many rows are sampled
        # for the animation by using the filtered dataset (not the display-limited one).
        timeline_sample_size = min(timeline_sample_size, len(filtered_data))
        with st.spinner("Rendering animated timeline..."):
            st.plotly_chart(
                create_animated_timeline(
                    filtered_data, sample_size=timeline_sample_size
                ),
                width='stretch',
            )
    except Exception as exc:
        st.error(f"Could not render the animated timeline: {exc}")

    st.subheader("🧊 3D Earthquake Visualization")
    st.caption(
        "3D plot showing longitude, latitude, depth, and magnitude relationships."
    )
    try:
        with st.spinner("Rendering 3D visualization..."):
            st.plotly_chart(
                create_3d_earthquake_visualization(
                    filtered_data, sample_size=timeline_sample_size
                ),
                width='stretch',
            )
    except Exception as exc:
        st.error(f"Could not render the 3D plot: {exc}")


def render_ml_tab(data):
    """Renders the 'ML Prediction' tab."""
    st.subheader("🤖 ML Trend Forecasting: Hybrid Model")
    st.caption(
        "A Hybrid Model (Exponential Smoothing + Gradient Boosting) analyzes monthly earthquake counts to reveal trends and complex patterns. "
        "This helps identify general earthquake frequency direction, not exact timing or locations."
    )

    # Show explanation first
    with st.expander("ℹ️ How does the Hybrid Model work?"):
        st.markdown(get_model_explanation())

    try:
        available_years = max(
            1,
            data["time"].dt.year.max() - data["time"].dt.year.min() + 1,
        )
        history_years = st.slider(
            "📆 Years of history to train on",
            min_value=1,
            max_value=available_years,
            value=available_years,
            help="Using more history usually improves stability. With your 2012+ data, the model can learn longer-term patterns.",
        )
        aggregation_label = st.radio(
            "Aggregation period",
            ["Monthly", "Weekly"],
            horizontal=True,
            help="Weekly data gives more samples, but monthly data is usually smoother and easier to predict.",
        )
        period = "M" if aggregation_label == "Monthly" else "W"

        # Get ML data from full history
        with st.spinner("Loading historical earthquake data..."):
            ml_data = get_ml_data_from_full_history(data, years=history_years)

        # Prepare time series data with the selected aggregation
        with st.spinner(
            f"Preparing {aggregation_label.lower()} earthquake frequency data..."
        ):
            frequency_data = prepare_time_series_data(ml_data, period=period)

        # Show model comparison summary
        with st.expander("🏆 Model Comparison (All Tested Models)"):
            st.caption(
                "Comparison of various ML models evaluated on the same earthquake frequency data. "
                "Ranked by Test R² score (higher is better)."
            )
            try:
                with st.spinner("Comparing models..."):
                    comparison_df = compare_models(frequency_data, period=period)
                if not comparison_df.empty:
                    # Format the dataframe for display
                    display_df = comparison_df.copy()
                    for col in [
                        "Accuracy (%)",
                        "Train R²",
                        "Test R²",
                        "Train RMSE",
                        "Test RMSE",
                        "Train MAE",
                        "Test MAE",
                    ]:
                        if col == "Accuracy (%)":
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                        else:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                            
                    # Reset index to show ranking
                    display_df.index = list(range(1, len(display_df) + 1))
                    display_df.index.name = "Rank"
                    
                    # Select only the relevant formatted columns to show in the table
                    st.table(display_df[["Model", "Accuracy (%)", "Train R²", "Test R²", "Test RMSE", "Test MAE"]])
                    
                    st.info("💡 **Why do simple models (like Baseline/Moving Average) sometimes show higher Accuracy?**\n\nEarthquake data is highly noisy. A Baseline model simply predicts a flat, average line. Mathematically, predicting a flat average is the 'safest' way to minimize percentage errors on chaotic data, which inflates its Accuracy score. However, a flat line is completely useless for forecasting. The Hybrid model sacrifices a tiny amount of raw accuracy because it actually attempts to capture the complex, underlying directional trends.")

                    st.markdown("---")
                    with st.container(border=True):
                        st.subheader("🔍 Detailed Model Statistics")
                        selected_model = st.selectbox("Select a model to view details", display_df["Model"].tolist())
                        
                        if selected_model:
                            model_stats = comparison_df[comparison_df["Model"] == selected_model].iloc[0]
                            det_col1, det_col2, det_col3, det_col4 = st.columns(4)
                            det_col1.metric("Accuracy", f"{model_stats['Accuracy (%)']:.2f}%")
                            det_col2.metric("Test R²", f"{model_stats['Test R²']:.4f}")
                            det_col3.metric("Test RMSE", f"{model_stats['Test RMSE']:.4f}")
                            det_col4.metric("Test MAE", f"{model_stats['Test MAE']:.4f}")
                            
                            st.divider()
                            
                            det_col5, det_col6, det_col7, det_col8 = st.columns(4)
                            det_col5.metric("Train R²", f"{model_stats['Train R²']:.4f}")
                            det_col6.metric("Train RMSE", f"{model_stats['Train RMSE']:.4f}")
                            det_col7.metric("Test MAPE", f"{model_stats['Test MAPE']:.2f}%")
                            det_col8.metric("Max Error", f"{model_stats['Test Max Error']:.2f}")

                            st.divider()
                            st.markdown("**Train vs Test Performance Comparison**")
                            
                            # Build a comparison chart
                            metrics_df = pd.DataFrame({
                                "Metric": ["R² Score", "RMSE", "MAE", "R² Score", "RMSE", "MAE"],
                                "Value": [
                                    max(0, model_stats["Train R²"]), model_stats["Train RMSE"], model_stats["Train MAE"],
                                    max(0, model_stats["Test R²"]), model_stats["Test RMSE"], model_stats["Test MAE"]
                                ],
                                "Dataset": ["Train", "Train", "Train", "Test", "Test", "Test"]
                            })
                            fig_comp = px.bar(
                                metrics_df, x="Metric", y="Value", color="Dataset", barmode="group",
                                color_discrete_map={"Train": "#94a3b8", "Test": "#0ea5a4"},
                                height=280
                            )
                            fig_comp.update_layout(
                                margin=dict(l=0, r=0, t=10, b=0), 
                                template="plotly_white",
                                yaxis_title="Score / Error",
                                xaxis_title=None
                            )
                            st.plotly_chart(fig_comp, width='stretch')
                            
                            # Add an automated generalization analysis
                            st.markdown("**Generalization Analysis:**")
                            if model_stats["Test R²"] < 0:
                                st.info("ℹ️ **Chaotic Data Expected:** Test R² is negative, which is mathematically common for earthquakes. A massive random earthquake swarm in the test data skews the test mean, making normal trend predictions score lower. The model correctly ignores these unpredictable spikes to maintain a stable baseline trend.")
                            elif model_stats["Train R²"] - model_stats["Test R²"] > 0.3:
                                st.warning("⚠️ **Overfitting Detected:** The model performs significantly better on training data than unseen test data. It may be memorizing noise instead of finding a true trend.")
                            else:
                                st.success("✅ **Good Generalization:** The model maintains balanced performance between training and test sets, making it reliable for extracting the underlying trend.")

                else:
                    st.warning("Could not run model comparison.")
            except Exception as e:
                st.info(f"Model comparison not available: {e}")

        # Check if we have enough data
        if len(frequency_data) < 15:
            st.warning(
                "Not enough data for ML model training. Need at least 15 time periods."
            )
        else:
            # Train the model
            with st.spinner("Training Hybrid Model..."):
                ml_results = train_ml_model(
                    frequency_data, test_size=0.2, period=period
                )

            # Display data summary
            st.markdown("### 📊 Data Summary")
            summary_cols = st.columns(4)
            summary_cols[0].metric(
                "Total Samples",
                ml_results["n_total_samples"],
                help="Total months of data used",
            )
            summary_cols[1].metric(
                "Training Samples",
                ml_results["n_train_samples"],
                help="80% of total data",
            )
            summary_cols[2].metric(
                "Testing Samples",
                ml_results["n_test_samples"],
                help="20% of total data",
            )
            summary_cols[3].metric(
                "Data Period",
                f"{aggregation_label}, {history_years} Yrs",
                help="Global historical earthquake data used for training",
            )

            # Display model performance metrics
            st.markdown("### 📊 Model Performance Metrics")
            accuracy = max(0.0, 100.0 - ml_results['test_mape'])
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(
                "Accuracy (%)",
                f"{accuracy:.2f}%",
                help="Overall accuracy derived from Mean Absolute Percentage Error",
            )
            metric_col2.metric(
                "Test R² Score",
                f"{ml_results['test_r2']:.3f}",
                help="How well model predicts test data (0-1, higher is better)",
            )
            metric_col3.metric(
                "Test RMSE",
                f"{ml_results['test_rmse']:.2f}",
                help="Root Mean Squared Error (lower is better)",
            )

            metric_col4, metric_col5, metric_col6 = st.columns(3)
            metric_col4.metric(
                "Test MAE",
                f"{ml_results['test_mae']:.2f}",
                help="Mean Absolute Error (Average earthquakes off by per period)",
            )
            metric_col5.metric(
                "Test MAPE",
                f"{ml_results['test_mape']:.1f}%",
                help="Mean Absolute Percentage Error (Average % error)",
            )
            metric_col6.metric(
                "Max Error",
                f"{ml_results['test_max_error']:.1f}",
                help="The largest single error in the test period",
            )

            # Display actual vs predicted visualization
            st.markdown("### 📊 Actual vs Predicted Earthquake Frequency")
            st.caption("Prediction accuracy: How closely the model predicts actual earthquake counts. Points near the red line are highly accurate.")
            st.plotly_chart(plot_actual_vs_predicted(ml_results), width='stretch')

            # Display Categorized Confusion Matrix & Classification Report
            st.markdown("### 🗂️ Categorized Confusion Matrix & Classification Report")
            st.caption(
                f"To evaluate this regression model categorically, we converted the continuous {aggregation_label.lower()} "
                "earthquake counts into 'Low', 'Medium', and 'High' activity categories based on historical averages."
            )

            cm_col, cr_col = st.columns(2)
            with cm_col:
                st.plotly_chart(plot_confusion_matrix(ml_results), width='stretch')
            with cr_col:
                st.dataframe(get_classification_report_df(ml_results), width='stretch')

            # Display trend forecast
            st.markdown("### 🔮 Future Trend Forecast")
            st.caption(
                f"Gray dots = raw {aggregation_label.lower()} data (noisy) | Teal line = Hybrid Model trend | Pink dotted line = future forecast"
            )
            
            # Add slider to control chart clutter
            max_history = len(ml_results["y_train"]) + len(ml_results["y_test"])
            default_history = 24 if period == "M" else 52 # 2 years for monthly, 1 year for weekly
            default_history = min(default_history, max_history)
            
            time_unit = "months" if period == "M" else "weeks"
            display_history = st.slider(
                f"Historical data to show in chart ({time_unit})",
                min_value=min(12, max_history),
                max_value=max_history,
                value=default_history,
                help="Reduce this to declutter the chart and focus on recent trends."
            )

            st.plotly_chart(
                create_prediction_plotly(
                    ml_results, 
                    future_periods=12 if period == "M" else 26,
                    display_history_periods=display_history
                ),
                width='stretch',
            )

            # ── Country-level Prediction Heatmap ─────────────────────────────
            st.markdown("### 🌍 Predicted Earthquakes by Country")
            st.caption(
                "Each country's historical monthly earthquake counts are modeled with "
                "Exponential Smoothing to forecast the selected number of months ahead. "
                "Countries with fewer than 18 months of data are excluded."
            )
            country_months = st.slider(
                "Forecast horizon (months)",
                min_value=3, max_value=24, value=12, step=3,
                key="country_forecast_months",
            )
            with st.spinner("Predicting earthquakes per country..."):
                try:
                    pred_df = predict_earthquakes_by_country(
                        data, future_months=country_months
                    )
                    if pred_df.empty:
                        st.warning("Not enough per-country data to build predictions.")
                    else:
                        top_country = pred_df.iloc[0]
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Countries Modeled", len(pred_df))
                        mc2.metric(
                            "🔴 Highest Risk",
                            top_country["country"],
                            f"{int(top_country['predicted_total'])} quakes",
                        )
                        mc3.metric(
                            "📈 Trending Up",
                            int((pred_df["trend"] == "↑ Up").sum()),
                            help="Countries with rising predicted activity",
                        )
                        mc4.metric(
                            "📉 Trending Down",
                            int((pred_df["trend"] == "↓ Down").sum()),
                            help="Countries with declining predicted activity",
                        )
                        st.plotly_chart(
                            plot_country_prediction_heatmap(pred_df, raw_data=data),
                            width='stretch',
                        )
                        st.markdown("#### 📋 Country Prediction Details")
                        display_pred = pred_df.copy()
                        display_pred.index = range(1, len(display_pred) + 1)
                        display_pred.columns = [
                            "Country", "Hist. Monthly Avg",
                            f"Pred. Total ({country_months}mo)",
                            "Pred. Monthly Avg", "Trend", "Confidence", "% Change"
                        ]
                        st.dataframe(display_pred, width='stretch')
                except Exception as country_exc:
                    st.error(f"Could not generate country predictions: {country_exc}")

            # Display important limitations
            st.markdown("### ⚠️ Important Limitations")
            st.warning("""
                **Earthquake activity is highly chaotic and difficult to predict accurately.**
                
                - This model only estimates **general trends**, not specific earthquakes
                - It captures patterns in historical data, but earthquakes are largely random
                - Regional data quality varies - some areas have more records than others
                - External factors (tectonic shifts, instrumentation changes) are not included
                - Earthquake frequency changes may be too irregular to predict with a simple model
                
                **Use this model to understand trends, NOT to predict when earthquakes will occur.**
                """)

    except Exception as exc:
        st.error(f"Could not train ML model: {exc}")



def render_about_tab():
    """Renders the 'About' tab with markdown information."""
    st.markdown("""
        ## About This Dashboard

        This dashboard uses the processed earthquake dataset from `data/historical_processed.csv`.

        ### Key Features
        - 🔍 **Advanced Filtering**: Filter by country, region, magnitude range, depth, and date
        - 🗺️ **Interactive Maps**: Heatmaps, magnitude-based, depth-based, and country overview maps
        - 📊 **Statistical Charts**: Distribution analysis, correlations, trends, and more
        - 🎬 **Animated Views**: Watch earthquakes unfold over time in 2D and 3D
        - 📈 **Real-time Analysis**: Instant calculations of statistics and categories

        ### Dataset Columns
        - `time`: When the earthquake occurred
        - `country`: Country where the earthquake occurred
        - `region`: Geographic region
        - `mag`: Magnitude (strength) of the earthquake
        - `depth`: Depth of the epicenter in kilometers
        - `latitude`/`longitude`: Geographic coordinates
        - `place`: Named location description
        """)

    with st.expander("How the filters work"):
        st.markdown("""
        - **Country**: Select a specific country or view all earthquakes worldwide
        - **Region**: Choose a specific region within a country (only available after selecting a country)
        - **Magnitude Range**: Focus on earthquakes within a specific strength range
        - **Depth Range**: Filter by how deep the earthquakes were
        - **Date Range**: Select a time period of interest
        """)

    with st.expander("What the visualizations show"):
        st.markdown("""
        - **Heatmap**: Shows density of earthquakes - brighter areas have more activity
        - **Magnitude Map**: Circle size and color intensity correspond to earthquake strength
        - **Depth Map**: Color gradient from light (shallow) to dark (very deep)
        - **Marker Clusters**: Clickable clusters that expand as you zoom in
        - **3D Visualization**: Shows how magnitude, location, and depth interact
        - **Animated Timeline**: Tracks how earthquake activity changes month by month
        - **Statistical Charts**: Distribution, correlations, and temporal trends
        """)

    with st.expander("Map Layer Information"):
        st.markdown("""
        - **CartoDB positron**: Light, clean map style
        - **CartoDB voyager**: Satellite-like imagery
        - **OpenStreetMap**: Classic street map view
        """)


def main():
    """Run the Streamlit dashboard."""

    apply_page_style()

    st.markdown(
        """
        <div class="title-block">
            <h1>🌍 Earthquake Data Analyzer</h1>
            <span class="badge">LIVE ANALYSIS</span>
            <span class="badge" style="background:linear-gradient(90deg,#7c3aed,#a78bfa);">ML POWERED</span>
            <p class="subtitle" style="margin-top:0.6rem;">
                Explore global seismic activity — filter by country, magnitude, depth &amp; time.
                Powered by a Hybrid ML model with country-level predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_data()

    if data.empty:
        st.error(
            "The processed dataset could not be loaded. Make sure data/historical_processed.csv exists."
        )
        st.stop()

    min_date = data["time"].dt.date.min()
    max_date = data["time"].dt.date.max()
    # default to the most recent 3 months (90 days) for the initial view
    default_start = (pd.to_datetime(max_date) - pd.Timedelta(days=90)).date()
    if default_start < min_date:
        default_start = min_date
    country_options = sorted(data["country"].dropna().unique().tolist())
    min_magnitude = float(data["mag"].min())
    max_magnitude = float(data["mag"].max())

    with st.sidebar:
        st.header("🔍 Filters")
        st.caption("Customize your view with these powerful filter options.")

        country_choice = st.selectbox("🌐 Country", ["All Countries"] + country_options)

        # Get regions for the selected country
        if country_choice == "All Countries":
            region_options = sorted(data["region"].dropna().unique().tolist())
        else:
            region_options = sorted(
                data[data["country"] == country_choice]["region"]
                .dropna()
                .unique()
                .tolist()
            )

        region_choice = (
            st.selectbox("🗺️ Region", ["All Regions"] + region_options)
            if region_options
            else "All Regions"
        )

        magnitude_range = st.slider(
            "📊 Magnitude Range",
            min_value=min_magnitude,
            max_value=max_magnitude,
            value=(min_magnitude, max_magnitude),
            key="magnitude_slider",
        )

        # Add depth filter
        min_depth = float(data["depth"].min())
        max_depth = float(data["depth"].max())
        depth_range = st.slider(
            "🔻 Depth Range (km)",
            min_value=min_depth,
            max_value=max_depth,
            value=(min_depth, max_depth),
            key="depth_slider",
        )

        selected_dates = st.date_input(
            "📅 Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
            date_range = tuple(selected_dates)
        else:
            date_range = (selected_dates, selected_dates)

        # Pre-calculate filtered data to dynamically adjust slider limits
        filtered_data = filter_data(
            data,
            country_choice,
            magnitude_range,
            date_range,
            depth_range,
            region_choice,
        )

        current_max = len(filtered_data)

        # Limit how many records to use for heavy visuals (maps, large charts)
        # Conditionally display slider to prevent min_value == max_value error
        if current_max > 100:
            max_records = st.slider(
                "Max records used for visuals",
                min_value=100,
                max_value=current_max,
                value=min(1000, current_max),
                step=100,
                help="Limits the sampled rows used by maps and large charts to keep the UI responsive.",
                key="max_records_slider",
            )
        else:
            max_records = current_max

        st.divider()
        st.subheader("Map Options")

        show_heatmap = st.checkbox("🔥 Show Heatmap", value=True)
        show_magnitude_map = st.checkbox("📍 Show Magnitude-based Map", value=False)
        show_depth_map = st.checkbox("🌊 Show Depth-based Map", value=False)
        show_cluster_map = st.checkbox("🎯 Show Marker Clusters", value=True)
        show_country_map = st.checkbox("🌍 Show Country Overview", value=False)

        st.divider()
        st.subheader("Chart Options")

        # Conditionally display slider to prevent min_value == max_value error
        if current_max > 100:
            timeline_sample_limit = min(10000, current_max)
            timeline_sample_size = st.slider(
                "Animated Timeline Sample Size",
                min_value=100,
                max_value=timeline_sample_limit,
                value=100,
                step=100,
                key="timeline_sample_slider",
            )
        else:
            timeline_sample_size = current_max

    if filtered_data.empty:
        st.warning(
            "No earthquakes matched the selected filters. Please widen the filter range."
        )
        st.stop()

    # Sample the filtered data for heavy visuals if it exceeds the user-selected max
    if len(filtered_data) > max_records:
        # use the most recent records to keep timelines and previews chronological
        display_data = (
            filtered_data.sort_values("time", ascending=False).head(max_records).copy()
        )
    else:
        display_data = filtered_data.copy()

    show_metric_cards(filtered_data)

    st.markdown("### 📋 Dataset Preview")
    preview_df = (
        filtered_data.sort_values("time", ascending=False)
        .copy()
        .loc[
            :,
            [
                "time",
                "country",
                "region",
                "mag",
                "depth",
                "latitude",
                "longitude",
                "place",
            ],
        ]
    )
    # Format time for readability in the UI preview
    preview_df["time"] = (
        pd.to_datetime(preview_df["time"])
        .dt.tz_convert(None)
        .dt.strftime("%Y-%m-%d %H:%M UTC")
    )
    st.dataframe(preview_df.head(20), width='stretch')

    st.markdown("### 📊 Quick Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.write(f"**Countries in view:** {filtered_data['country'].nunique()}")
    summary_col2.write(f"**Regions in view:** {filtered_data['region'].nunique()}")
    min_time = pd.to_datetime(filtered_data["time"]).dt.tz_convert(None).min()
    max_time = pd.to_datetime(filtered_data["time"]).dt.tz_convert(None).max()
    summary_col3.write(
        f"**Date span:** {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}"
    )

    charts_tab, maps_tab, advanced_tab, animation_tab, ml_tab, about_tab = st.tabs(
        [
            "📊 Charts",
            "🗺️ Maps",
            "🎬 Advanced",
            "🔗 3D & Animation",
            "🤖 ML Prediction",
            "ℹ️ About",
        ]
    )

    with charts_tab:
        render_charts_tab(filtered_data)

    with maps_tab:
        map_options = {
            "show_heatmap": show_heatmap,
            "show_magnitude_map": show_magnitude_map,
            "show_depth_map": show_depth_map,
            "show_cluster_map": show_cluster_map,
            "show_country_map": show_country_map,
        }
        render_maps_tab(display_data, map_options)

    with advanced_tab:
        render_advanced_tab(filtered_data)

    with animation_tab:
        render_animation_tab(filtered_data, timeline_sample_size)

    with ml_tab:
        render_ml_tab(data)

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
