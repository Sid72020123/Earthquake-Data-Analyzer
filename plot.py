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


import folium
from folium.plugins import MarkerCluster
import pandas as pd

df = pd.read_csv("data/historical_processed.csv")

# Base map
m = folium.Map(location=[20, 0], zoom_start=2)

# Create cluster layer
marker_cluster = MarkerCluster().add_to(m)

# Add markers
for _, row in df[:1000].iterrows():

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"""
        Country: {row['country']}<br>
        Magnitude: {row['mag']}<br>
        Depth: {row['depth']}
        """,
    ).add_to(marker_cluster)

# Save map
m.save("earthquake_cluster.html")
