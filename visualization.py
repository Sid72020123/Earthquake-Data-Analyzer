import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import folium
from branca.colormap import LinearColormap

from folium.plugins import HeatMap, MarkerCluster

# Apply plotting theme based on Streamlit theme (light/dark) when available.
try:
    theme_base = st.get_option("theme.base")
except Exception:
    theme_base = "light"

if theme_base == "dark":
    sns.set_style("darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1220",
            "axes.facecolor": "#0b1220",
            "savefig.facecolor": "#0b1220",
            "text.color": "#e6eef8",
            "axes.labelcolor": "#e6eef8",
            "xtick.color": "#e6eef8",
            "ytick.color": "#e6eef8",
        }
    )
else:
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "text.color": "#0f172a",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#0f172a",
            "ytick.color": "#0f172a",
        }
    )


def _empty_matplotlib_figure(message):
    """Create a small figure that explains why no chart is shown."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return fig


def plot_magnitude_distribution(df):
    """Show how earthquake magnitudes are spread across the dataset."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the histogram.")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["mag"], bins=30, kde=True, ax=ax, color="#c2410c")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Magnitude Distribution")
    plt.tight_layout()
    return fig


def plot_depth_vs_magnitude(df):
    """Compare earthquake depth and magnitude with a scatter plot."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the scatter plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="depth", y="mag", data=df, alpha=0.5, ax=ax, color="#0369a1")
    ax.set_title("Depth vs Magnitude")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Magnitude")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Show a simple correlation heatmap for the numeric columns."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the heatmap.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[["mag", "depth", "latitude", "longitude"]].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_top_countries_bar_chart(df, top_n=10):
    """Show the countries with the most earthquakes."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the country chart.")

    top_countries = df["country"].value_counts().head(top_n).reset_index()
    top_countries.columns = ["country", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="country", y="count", data=top_countries, ax=ax, color="#0f766e")
    ax.set_title(f"Top {top_n} Earthquake Countries")
    ax.set_xlabel("Country")
    ax.set_ylabel("Earthquake Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_top_countries_pie_chart(df, top_n=5):
    """Show the top countries as a simple pie chart."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the pie chart.")

    top_countries = df["country"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(top_countries, labels=top_countries.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Top {top_n} Earthquake Countries")
    return fig


def plot_earthquake_trend(df):
    """Plot earthquake counts over time using monthly counts."""

    if df.empty:
        return _empty_matplotlib_figure("No data available for the time-series chart.")

    time_df = df.copy()
    time_df["time"] = pd.to_datetime(time_df["time"], errors="coerce", utc=True)
    time_df = time_df.dropna(subset=["time"])
    time_df["month"] = (
        time_df["time"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    )
    monthly_counts = time_df.groupby("month").size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=monthly_counts, x="month", y="count", marker="o", ax=ax, color="#7c2d12"
    )
    ax.set_title("Earthquake Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Earthquake Count")
    plt.tight_layout()
    return fig


def create_folium_heatmap(df):
    """Create a Folium heatmap for the earthquake locations."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    if df.empty:
        return m

    heat_data = df[["latitude", "longitude", "mag"]].dropna().values.tolist()
    HeatMap(heat_data, radius=8, blur=12, max_zoom=5).add_to(m)
    return m


def create_marker_cluster_map(df, sample_size=1000):
    """Create a simple marker cluster map for a sample of earthquakes."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    if df.empty:
        return m

    map_data = df.dropna(subset=["latitude", "longitude", "mag", "country"]).head(
        sample_size
    )
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in map_data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(float(row["mag"]), 1.5),
            color="#b91c1c",
            fill=True,
            fill_opacity=0.6,
            popup=f"Country: {row['country']}<br>Magnitude: {row['mag']}<br>Depth: {row['depth']}<br>Time: {pd.to_datetime(row.get('time')).strftime('%Y-%m-%d %H:%M UTC')}",
        ).add_to(marker_cluster)

    return m


def create_magnitude_based_map(df, sample_size=2000):
    """Create a folium map with magnitude-based coloring and sizing."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB voyager")

    if df.empty:
        return m

    map_data = df.dropna(subset=["latitude", "longitude", "mag", "country"]).head(
        sample_size
    )

    # Create a color scale based on magnitude
    min_mag = map_data["mag"].min()
    max_mag = map_data["mag"].max()
    colormap = LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"],
        vmin=min_mag,
        vmax=max_mag,
        caption="Magnitude",
    )

    for _, row in map_data.iterrows():
        mag = float(row["mag"])
        radius = max(2 + mag * 1.5, 3)
        color = colormap(mag)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            popup=folium.Popup(
                f"""<b>{row['country']}</b><br>
                Magnitude: {row['mag']}<br>
                Depth: {row['depth']} km<br>
                Time: {pd.to_datetime(row['time']).strftime('%Y-%m-%d %H:%M UTC')}<br>
                Region: {row.get('region', 'N/A')}""",
                max_width=250,
            ),
            tooltip=f"{row['country']} - Mag: {row['mag']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    m.add_child(colormap)
    return m


def create_depth_based_map(df, sample_size=2000):
    """Create a folium map with depth-based coloring."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    if df.empty:
        return m

    map_data = df.dropna(subset=["latitude", "longitude", "depth", "country"]).head(
        sample_size
    )

    # Create a color scale based on depth
    min_depth = map_data["depth"].min()
    max_depth = map_data["depth"].max()
    colormap = LinearColormap(
        colors=["#d4ee31", "#90ee90", "#20b2aa", "#4169e1", "#00008b"],
        vmin=min_depth,
        vmax=max_depth,
        caption="Depth (km)",
    )

    for _, row in map_data.iterrows():
        depth = float(row["depth"])
        mag = float(row.get("mag", 4))
        radius = max(2 + mag, 3)
        color = colormap(depth)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            popup=folium.Popup(
                f"""<b>{row['country']}</b><br>
                Depth: {row['depth']} km<br>
                Magnitude: {mag}<br>
                Time: {pd.to_datetime(row['time']).strftime('%Y-%m-%d %H:%M UTC')}<br>
                Region: {row.get('region', 'N/A')}""",
                max_width=250,
            ),
            tooltip=f"{row['country']} - Depth: {row['depth']} km",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    m.add_child(colormap)
    return m


def create_country_region_map(df):
    """Create a map with country boundaries highlighted and earthquake data."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")

    if df.empty:
        return m

    # Create a heatmap first
    heat_data = df[["latitude", "longitude", "mag"]].dropna().values.tolist()
    HeatMap(heat_data, radius=10, blur=15, max_zoom=4, min_opacity=0.3).add_to(m)

    # Add some markers for visual richness
    for country, group in df.groupby("country"):
        if len(group) > 0:
            center_lat = group["latitude"].mean()
            center_lon = group["longitude"].mean()
            count = len(group)
            avg_mag = group["mag"].mean()

            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=5 + avg_mag,
                popup=folium.Popup(
                    f"""<b>{country}</b><br>
                    Earthquakes: {count}<br>
                    Avg Magnitude: {avg_mag:.2f}""",
                    max_width=200,
                ),
                tooltip=f"{country}: {count} earthquakes",
                color="#1f77b4",
                fill=True,
                fillColor="#1f77b4",
                fillOpacity=0.6,
                weight=2,
            ).add_to(m)

    return m


def create_animated_timeline(df, sample_size=5000):
    """Create a Plotly animated earthquake timeline."""

    if df.empty:
        return px.scatter_geo(title="No data available for the animated timeline")

    timeline_df = df.copy().dropna(subset=["time", "latitude", "longitude", "mag"])
    if len(timeline_df) > sample_size:
        timeline_df = timeline_df.sample(sample_size, random_state=42)

    timeline_df["time"] = pd.to_datetime(timeline_df["time"], errors="coerce", utc=True)
    timeline_df = timeline_df.dropna(subset=["time"])

    timeline_df["month_ts"] = (
        timeline_df["time"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    )
    timeline_df["month"] = timeline_df["month_ts"].dt.strftime("%Y-%m")
    month_order = sorted(timeline_df["month"].unique())
    timeline_df["month"] = pd.Categorical(
        timeline_df["month"], categories=month_order, ordered=True
    )

    # Format hover time for readability
    timeline_df["time_str"] = timeline_df["time"].dt.strftime("%Y-%m-%d %H:%M UTC")

    try:
        base_theme = st.get_option("theme.base")
    except Exception:
        base_theme = "light"

    plotly_template = "plotly_dark" if base_theme == "dark" else "plotly_white"

    fig = px.scatter_geo(
        timeline_df,
        lat="latitude",
        lon="longitude",
        color="mag",
        size="mag",
        hover_name="country",
        hover_data={"time_str": True, "mag": True},
        animation_frame="month",
        projection="natural earth",
        title="Animated Earthquake Timeline",
        color_continuous_scale="YlOrRd",
        height=650,
        template=plotly_template,
        category_orders={"month": month_order},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        transition={"duration": 400, "easing": "linear"},
    )
    return fig


def create_3d_earthquake_visualization(df, sample_size=5000):
    """Create a simple 3D scatter plot for location, depth, and magnitude."""

    if df.empty:
        return px.scatter_3d(title="No data available for the 3D plot")

    plot_df = df.copy().dropna(
        subset=["latitude", "longitude", "depth", "mag", "country"]
    )
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    fig = px.scatter_3d(
        plot_df,
        x="longitude",
        y="latitude",
        z="depth",
        color="mag",
        hover_name="country",
        title="3D Earthquake Visualization",
        color_continuous_scale="Turbo",
        height=650,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig
