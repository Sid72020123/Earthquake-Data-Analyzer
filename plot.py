# import folium
# from folium.plugins import HeatMap
# import pandas as pd

# df = pd.read_csv("data/historical_processed.csv")

# print(df.head())

# m = folium.Map(location=[20, 0], zoom_start=2)

# for _, row in df[0:20].iterrows():
#     folium.CircleMarker(
#         location=[row["latitude"], row["longitude"]],
#         radius=row["mag"],
#         popup=f"""
#         Country: {row['country']}<br>
#         Magnitude: {row['mag']}<br>
#         Depth: {row['depth']}
#         """,
#         fill=True,
#     ).add_to(m)

# m.save("earthquake_map.html")


# import folium
# from folium.plugins import HeatMap
# import pandas as pd

# df = pd.read_csv("data/historical_processed.csv")

# # Create base map
# m = folium.Map(location=[20, 0], zoom_start=2)

# # Prepare heatmap data
# heat_data = df[["latitude", "longitude", "mag"]].values.tolist()

# # Add heatmap
# # HeatMap(heat_data).add_to(m)
# # HeatMap(heat_data, radius=10, blur=15, max_zoom=5).add_to(m)
# HeatMap(heat_data, radius=8, blur=12).add_to(m)

# # Save map
# m.save("earthquake_heatmap.html")


"""Optional standalone script for exporting a simple earthquake marker cluster map."""

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
