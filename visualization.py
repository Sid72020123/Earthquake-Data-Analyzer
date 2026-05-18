import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


def plot_magnitude_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(df["mag"], bins=30, kde=True)

    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude Distribution")

    plt.show()


def create_heatmap(df):

    m = folium.Map(location=[20, 0], zoom_start=2)

    heat_data = df[["latitude", "longitude", "mag"]].values.tolist()

    HeatMap(heat_data).add_to(m)

    return m


def plot_depth_vs_magnitude(df):

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x="depth", y="mag", data=df)

    plt.title("Depth vs Magnitude")

    plt.show()


def plot_correlation_heatmap(df):

    plt.figure(figsize=(8, 6))

    sns.heatmap(df[["mag", "depth", "latitude", "longitude"]].corr(), annot=True)

    plt.title("Correlation Heatmap")

    plt.show()


def plot_magnitude_boxplot(df):

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df["mag"])

    plt.title("Magnitude Boxplot")

    plt.show()


def plot_country_counts(df):

    top_countries = df["country"].value_counts().head(10).reset_index()

    top_countries.columns = ["country", "count"]

    plt.figure(figsize=(12, 6))

    sns.barplot(x="country", y="count", data=top_countries)

    plt.xticks(rotation=45)

    plt.title("Top Earthquake Countries")

    plt.show()
