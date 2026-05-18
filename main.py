import pandas as pd
from visualization import (
    plot_magnitude_distribution,
    create_heatmap,
    plot_country_counts,
    plot_depth_vs_magnitude,
    plot_correlation_heatmap,
    plot_magnitude_boxplot,
    plot_country_counts,
    create_animated_earthquake_map,
    create_3d_earthquake_plot
)

df = pd.read_csv("data/historical_processed.csv")
print(df.info())
# plot_magnitude_distribution(df)
# plot_country_counts(df)
# plot_depth_vs_magnitude(df)
# plot_correlation_heatmap(df)
# plot_magnitude_boxplot(df)
# plot_country_counts(df)
# create_animated_earthquake_map(df).show()
create_3d_earthquake_plot.show()

"""
# print(df.info())
# print(df.describe())

print("Total Earthquakes:", len(df))
print("Mean Magnitude:", df["mag"].mean())
print("Maximum Magnitude:", df["mag"].max())
print("Minimum Magnitude:", df["mag"].min())
print("Median Magnitude:", df["mag"].median())
print("Standard Deviation:", df["mag"].std())

# Count earthquakes per region:
# print(df["region"].value_counts())

# # Top 10 regions:
# print(df["region"].value_counts().head(10))

# # Average Magnitude by Region
# print(df.groupby("region")["mag"].mean())

# Strongest Earthquake:
print(df.loc[df["mag"].idxmax()])

# Deepest Earthquake
print(df.loc[df["depth"].idxmax()])

# Earthquake Frequency over Time
daily_counts = df.groupby(df["time"].dt.date).size()

print(daily_counts)

# Monthly Counts
monthly_counts = df.groupby(df["time"].dt.month).size()

print(monthly_counts)

# Correlation Analysis
# This checks: whether deeper earthquakes tend to be stronger.

print(df[["mag", "depth"]].corr())


# Magnitude Distribution Counts by bins
bins = [0, 2, 4, 6, 8, 10]

print(pd.cut(df["mag"], bins).value_counts())


print(df["region"].unique())



print(df.describe())
"""
