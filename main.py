"""Streamlit dashboard for the Earthquake Data Analyzer project."""

import pandas as pd
import streamlit as st
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
    plot_top_countries_pie_chart,
)

from ml_prediction import (
    prepare_time_series_data,
    train_ml_model,
    plot_actual_vs_predicted,
    create_prediction_plotly,
    create_feature_importance_plot,
    get_model_explanation,
    get_ml_data_from_full_history,
    compare_models,
)

DATA_PATH = "data/historical_processed.csv"


st.set_page_config(
    page_title="Earthquake Data Analyzer",
    page_icon="🌍",
    layout="wide",
)


def apply_page_style():
    """Add a small custom style so the page looks cleaner."""

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
            }
            .title-block {
                padding: 1.5rem 1.25rem;
                background: #ffffff;
                border-radius: 1.25rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
                margin-bottom: 1rem;
            }
            .title-block h1 {
                margin-bottom: 0.25rem;
                color: #0f172a;
            }
            .title-block p {
                margin-bottom: 0;
                color: #475569;
                font-size: 1rem;
            }
            /* Typography and accent */
            .stApp {
                font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            }
            .title-block {
                border-left: 6px solid #0ea5a4;
            }
            /* Dark mode adjustments */
            [data-theme="dark"] .stApp {
                background: linear-gradient(180deg, #071027 0%, #0b1220 100%);
            }
            [data-theme="dark"] .title-block {
                background: #0f172a;
                box-shadow: 0 10px 30px rgba(255, 255, 255, 0.03);
                border-left: 6px solid #06b6d4;
            }
            [data-theme="dark"] .title-block h1 {
                color: #e6eef8;
            }
            [data-theme="dark"] .title-block p {
                color: #cbd5e1;
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


def main():
    """Run the Streamlit dashboard."""

    apply_page_style()

    st.markdown(
        """
        <div class="title-block">
            <h1>Earthquake Data Analyzer</h1>
            <p>
                A beginner-friendly dashboard for exploring earthquake patterns by country,
                magnitude, depth, and time.
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
    # default to the most recent 15 days for the initial view
    default_start = (pd.to_datetime(max_date) - pd.Timedelta(days=14)).date()
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
        )

        # Add depth filter
        min_depth = float(data["depth"].min())
        max_depth = float(data["depth"].max())
        depth_range = st.slider(
            "🔻 Depth Range (km)",
            min_value=min_depth,
            max_value=max_depth,
            value=(min_depth, max_depth),
        )

        selected_dates = st.date_input(
            "📅 Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Limit how many records to use for heavy visuals (maps, large charts)
        # Start by showing only 100 records in visuals to keep the app responsive
        max_records_default = 100
        max_records = st.slider(
            "Max records used for visuals",
            min_value=100,
            max_value=max(100, len(data)),
            value=max_records_default,
            step=100,
            help="Limits the sampled rows used by maps and large charts to keep the UI responsive.",
        )

        st.divider()
        st.subheader("Map Options")

        show_heatmap = st.checkbox("🔥 Show Heatmap", value=True)
        show_magnitude_map = st.checkbox("📍 Show Magnitude-based Map", value=False)
        show_depth_map = st.checkbox("🌊 Show Depth-based Map", value=False)
        show_cluster_map = st.checkbox("🎯 Show Marker Clusters", value=True)
        show_country_map = st.checkbox("🌍 Show Country Overview", value=False)

        st.divider()
        st.subheader("Chart Options")

        # Keep timeline sampling small by default (100) and allow up to dataset size
        timeline_sample_limit = max(100, min(10000, len(data)))
        timeline_sample_size = st.slider(
            "Animated Timeline Sample Size",
            min_value=100,
            max_value=timeline_sample_limit,
            value=min(100, timeline_sample_limit),
            step=100,
        )

    if isinstance(selected_dates, tuple):
        date_range = selected_dates
    else:
        date_range = (selected_dates, selected_dates)

    filtered_data = filter_data(
        data, country_choice, magnitude_range, date_range, depth_range, region_choice
    )

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
    st.dataframe(preview_df.head(20), width="stretch")

    st.markdown("### 📊 Quick Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.write(f"**Countries in view:** {filtered_data['country'].nunique()}")
    summary_col2.write(f"**Regions in view:** {filtered_data['region'].nunique()}")
    min_time = pd.to_datetime(filtered_data["time"]).dt.tz_convert(None).min()
    max_time = pd.to_datetime(filtered_data["time"]).dt.tz_convert(None).max()
    summary_col3.write(
        f"**Date span:** {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}"
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Charts",
            "🗺️ Maps",
            "🎬 Advanced",
            "🔗 3D & Animation",
            "🤖 ML Prediction",
            "ℹ️ About",
        ]
    )

    with tab1:
        st.subheader("Statistical Analysis")
        left_col, right_col = st.columns(2)
        with left_col:
            st.pyplot(plot_magnitude_distribution(display_data), width="stretch")
            st.pyplot(plot_depth_vs_magnitude(display_data), width="stretch")
            st.pyplot(plot_earthquake_trend(display_data), width="stretch")
        with right_col:
            st.pyplot(plot_correlation_heatmap(display_data), width="stretch")
            st.pyplot(plot_top_countries_bar_chart(display_data), width="stretch")
            st.pyplot(plot_top_countries_pie_chart(display_data), width="stretch")

    with tab2:
        st.subheader("🔥 Folium Earthquake Heatmap")
        st.caption(
            "Dense earthquake locations are highlighted. Zoom in to explore specific areas."
        )
        if show_heatmap:
            try:
                with st.spinner("Rendering heatmap..."):
                    st_folium(
                        create_folium_heatmap(display_data), width=1200, height=600
                    )
            except Exception as exc:
                st.error(f"Could not render the heatmap: {exc}")
        else:
            st.info("Enable heatmap in the sidebar filters to view this map.")

        col_map1, col_map2 = st.columns(2)

        with col_map1:
            st.subheader("📍 Magnitude-based Map")
            st.caption("Circle size and color represent earthquake magnitude.")
            if show_magnitude_map:
                try:
                    with st.spinner("Rendering magnitude map..."):
                        st_folium(
                            create_magnitude_based_map(display_data),
                            width=600,
                            height=500,
                        )
                except Exception as exc:
                    st.error(f"Could not render the magnitude map: {exc}")
            else:
                st.info("Enable magnitude map in sidebar to view.")

        with col_map2:
            st.subheader("🌊 Depth-based Map")
            st.caption("Circle color represents depth (shallow to deep).")
            if show_depth_map:
                try:
                    with st.spinner("Rendering depth map..."):
                        st_folium(
                            create_depth_based_map(display_data), width=600, height=500
                        )
                except Exception as exc:
                    st.error(f"Could not render the depth map: {exc}")
            else:
                st.info("Enable depth map in sidebar to view.")

        st.subheader("🎯 Interactive Marker Cluster Map")
        st.caption(
            "Click clusters to zoom in. Individual markers appear at higher zoom levels."
        )
        if show_cluster_map:
            try:
                with st.spinner("Rendering cluster map..."):
                    st_folium(
                        create_marker_cluster_map(display_data), width=1200, height=600
                    )
            except Exception as exc:
                st.error(f"Could not render the cluster map: {exc}")
        else:
            st.info("Enable marker clusters in the sidebar filters to view this map.")

        st.subheader("🌍 Country Overview Map")
        st.caption(
            "Heatmap showing earthquake density across regions with country markers."
        )
        if show_country_map:
            try:
                with st.spinner("Rendering country overview map..."):
                    st_folium(
                        create_country_region_map(display_data), width=1200, height=600
                    )
            except Exception as exc:
                st.error(f"Could not render the country map: {exc}")
        else:
            st.info("Enable country overview map in sidebar to view.")

    with tab3:
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
        time_data["date"] = pd.to_datetime(time_data["time"]).dt.date
        daily_counts = time_data.groupby("date").size()
        st.line_chart(daily_counts)

    with tab4:
        st.subheader("🎬 Plotly Animated Timeline")
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
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Could not render the animated timeline: {exc}")

        st.subheader("🧊 Plotly 3D Earthquake Visualization")
        st.caption(
            "3D plot showing longitude, latitude, depth, and magnitude relationships."
        )
        try:
            with st.spinner("Rendering 3D visualization..."):
                st.plotly_chart(
                    create_3d_earthquake_visualization(
                        filtered_data, sample_size=timeline_sample_size
                    ),
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Could not render the 3D plot: {exc}")

    with tab5:
        st.subheader("🤖 ML Trend Forecasting: Moving Average Model")
        st.caption(
            "A Moving Average model smooths recent monthly earthquake counts to reveal short-term trends. "
            "This helps identify general earthquake frequency direction, not exact timing or locations."
        )

        # Show explanation first
        with st.expander("ℹ️ How does the Moving Average model work?"):
            st.markdown(get_model_explanation())

        try:
            # Get ML data from full history (last 5 years)
            with st.spinner("Loading historical earthquake data..."):
                ml_data = get_ml_data_from_full_history(data, years=5)

            # Prepare time series data (monthly frequency)
            with st.spinner("Preparing monthly earthquake frequency data..."):
                frequency_data = prepare_time_series_data(ml_data, period="M")

            # Show model comparison summary
            with st.expander("🏆 Model Comparison (All Tested Models)"):
                st.caption(
                    "Comparison of 7 different ML models evaluated on the same earthquake frequency data. "
                    "Ranked by Test R² score (higher is better)."
                )
                try:
                    with st.spinner("Comparing models..."):
                        comparison_df = compare_models(frequency_data)
                    if not comparison_df.empty:
                        # Format the dataframe for display
                        display_df = comparison_df.copy()
                        display_df["Train R²"] = display_df["Train R²"].apply(
                            lambda x: f"{x:.4f}"
                        )
                        display_df["Test R²"] = display_df["Test R²"].apply(
                            lambda x: f"{x:.4f}"
                        )
                        display_df["Train RMSE"] = display_df["Train RMSE"].apply(
                            lambda x: f"{x:.4f}"
                        )
                        display_df["Test RMSE"] = display_df["Test RMSE"].apply(
                            lambda x: f"{x:.4f}"
                        )
                        # Reset index to show ranking 1-7
                        display_df.index = list(range(1, len(display_df) + 1))
                        display_df.index.name = "Rank"
                        st.table(display_df)

                        # st.markdown(
                        #     f"**Selected Model:** {comparison_df.iloc[0]['Model']} "
                        #     f"(Test R² = {comparison_df.iloc[0]['Test R²']:.4f})"
                        # )
                    else:
                        st.warning("Could not run model comparison.")
                except Exception as e:
                    st.info(f"Model comparison not available: {e}")

            # Check if we have enough data
            if len(frequency_data) < 15:
                st.warning(
                    "Not enough data for ML model training. Need at least 15 months."
                )
            else:
                # Train the model
                with st.spinner("Training Moving Average model..."):
                    ml_results = train_ml_model(frequency_data, test_size=0.2)

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
                    "Last 5 Years",
                    help="Global historical earthquake data",
                )

                # Display model performance metrics
                st.markdown("### 📊 Model Performance Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric(
                    "Train R² Score",
                    f"{ml_results['train_r2']:.3f}",
                    help="How well model fits training data (0-1, higher is better)",
                )
                metric_col2.metric(
                    "Test R² Score",
                    f"{ml_results['test_r2']:.3f}",
                    help="How well model predicts test data (0-1, higher is better)",
                )
                metric_col3.metric(
                    "Test RMSE",
                    f"{ml_results['test_rmse']:.4f}",
                    help="Average prediction error (lower is better)",
                )

                # Display train RMSE as well
                st.markdown("### 📈 Training RMSE")
                st.write(
                    f"**Train RMSE:** {ml_results['train_rmse']:.4f} - Average error on training data"
                )

                # Display actual vs predicted visualization
                st.markdown("### 📊 Actual vs Predicted Earthquake Frequency")
                st.caption(
                    "Left: How the trend line compares to actual monthly data | Right: Prediction accuracy"
                )
                st.pyplot(plot_actual_vs_predicted(ml_results), width="stretch")

                # Display feature importance
                st.markdown("### 🎯 Feature Importance")
                st.caption("How important time is for the model's predictions")
                st.pyplot(create_feature_importance_plot(ml_results), width="stretch")

                # Display trend forecast
                st.markdown("### 🔮 12-Month Trend Forecast")
                st.caption(
                    "Gray dots = raw monthly data (noisy) | Teal line = Moving Average trend | Pink dotted line = future forecast"
                )
                st.plotly_chart(
                    create_prediction_plotly(ml_results, future_periods=12),
                    width="stretch",
                )

                # Display important limitations
                st.markdown("### ⚠️ Important Limitations")
                st.warning("""
                    **Earthquake activity is highly chaotic and difficult to predict accurately.**
                    
                    - This model only estimates **general trends**, not specific earthquakes
                    - It captures patterns in historical data, but earthquakes are largely random
                    - Regional data quality varies - some areas have better records than others
                    - External factors (tectonic shifts, instrumentation changes) are not included
                    - Earthquake frequency changes may be too irregular to predict with a simple model
                    
                    **Use this model to understand trends, NOT to predict when earthquakes will occur.**
                    """)

        except Exception as exc:
            st.error(f"Could not train ML model: {exc}")
            import traceback

            st.error(traceback.format_exc())

    with tab6:
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


if __name__ == "__main__":
    main()
