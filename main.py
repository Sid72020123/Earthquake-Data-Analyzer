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
            /* Dark mode adjustments */
            [data-theme="dark"] .stApp {
                background: linear-gradient(180deg, #071027 0%, #0b1220 100%);
            }
            [data-theme="dark"] .title-block {
                background: #0f172a;
                box-shadow: 0 10px 30px rgba(255, 255, 255, 0.03);
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
        data = pd.read_csv(DATA_PATH)
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
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Limit how many records to use for heavy visuals (maps, large charts)
        max_records_default = min(5000, len(data)) if len(data) > 0 else 500
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

        timeline_sample_limit = max(500, min(10000, len(data)))
        timeline_sample_size = st.slider(
            "Animated Timeline Sample Size",
            min_value=500,
            max_value=timeline_sample_limit,
            value=min(5000, timeline_sample_limit),
            step=500,
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
        display_data = filtered_data.sample(n=max_records, random_state=42)
    else:
        display_data = filtered_data

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Charts", "🗺️ Maps", "🎬 Advanced", "🔗 3D & Animation", "ℹ️ About"]
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
                st_folium(create_folium_heatmap(display_data), width=1200, height=600)
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
                    st_folium(
                        create_magnitude_based_map(display_data), width=600, height=500
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
            timeline_sample_size = min(timeline_sample_size, max_records)
            st.plotly_chart(
                create_animated_timeline(
                    display_data, sample_size=timeline_sample_size
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
            st.plotly_chart(
                create_3d_earthquake_visualization(
                    display_data, sample_size=timeline_sample_size
                ),
                width="stretch",
            )
        except Exception as exc:
            st.error(f"Could not render the 3D plot: {exc}")

    with tab5:
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
