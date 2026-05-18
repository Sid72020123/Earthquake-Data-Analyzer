import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

from folium.plugins import HeatMap

# Better default styling
sns.set_style("whitegrid")


# 1. Magnitude Distribution
def plot_magnitude_distribution(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(df["mag"], bins=30, kde=True, ax=ax)

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Magnitude Distribution")

    plt.tight_layout()

    return fig


# 2. Earthquake HeatMap
def create_heatmap(df):

    m = folium.Map(location=[20, 0], zoom_start=2)

    heat_data = df[["latitude", "longitude", "mag"]].values.tolist()

    HeatMap(heat_data, radius=8, blur=12, max_zoom=5).add_to(m)

    return m


# 3. Depth vs Magnitude
def plot_depth_vs_magnitude(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x="depth", y="mag", data=df, alpha=0.5, ax=ax)

    ax.set_title("Depth vs Magnitude")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Magnitude")

    plt.tight_layout()

    return fig


# 4. Correlation Heatmap
def plot_correlation_heatmap(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df[["mag", "depth", "latitude", "longitude"]].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax,
    )

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()

    return fig


# 5. Magnitude Boxplot
def plot_magnitude_boxplot(df):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(x=df["mag"], ax=ax)

    ax.set_title("Magnitude Boxplot")

    plt.tight_layout()

    return fig


# 6. Top Earthquake Countries
def plot_country_counts(df):

    top_countries = df["country"].value_counts().head(10).reset_index()

    top_countries.columns = ["country", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x="country", y="count", data=top_countries, ax=ax)

    ax.set_title("Top Earthquake Countries")
    ax.set_xlabel("Country")
    ax.set_ylabel("Earthquake Count")

    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


# 7. Time-Series Plot
def plot_earthquakes_over_time(df):

    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

    daily_counts = df.groupby(df["time"].dt.date).size()

    fig, ax = plt.subplots(figsize=(14, 6))

    daily_counts.plot(ax=ax)

    ax.set_title("Earthquakes Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Earthquake Count")

    plt.tight_layout()

    return fig


def plot_country_pie_chart(df):

    top_countries = df["country"].value_counts().head(5)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(top_countries, labels=top_countries.index, autopct="%1.1f%%")

    ax.set_title("Top 5 Earthquake Countries")

    return fig


def create_animated_earthquake_map(df):
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    df["date"] = df["time"].dt.date

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="mag",
        size="mag",
        hover_name="country",
        animation_frame=df["date"].astype(str),
        projection="natural earth",
        title="Earthquakes Over Time",
    )

    return fig
