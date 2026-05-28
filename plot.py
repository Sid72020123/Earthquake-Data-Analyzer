"""
Optional standalone script for exporting a simple earthquake marker cluster map.

NOTE: This file is kept for reference only. All plotting functionality has been
migrated to visualization.py and main.py.
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster

DATA_PATH = "data/historical_processed.csv"
OUTPUT_PATH = "earthquake_cluster.html"


def create_cluster_map(df, sample_size=1000):
    """Create a simple marker cluster map from the processed dataset."""

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    marker_cluster = MarkerCluster().add_to(m)

    map_data = df.dropna(subset=["latitude", "longitude", "mag", "country"]).head(sample_size)

    for _, row in map_data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(float(row["mag"]), 1.5),
            color="#b91c1c",
            fill=True,
            fill_opacity=0.6,
            popup=f"Country: {row['country']}<br>Magnitude: {row['mag']}<br>Depth: {row['depth']}",
        ).add_to(marker_cluster)

    return m


if __name__ == "__main__":
    try:
        earthquakes = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Could not find {DATA_PATH}.")
    else:
        cluster_map = create_cluster_map(earthquakes)
        cluster_map.save(OUTPUT_PATH)
        print(f"Saved {OUTPUT_PATH}")
