import os

import pandas as pd
import reverse_geocoder as rg
import pycountry

ROOT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"


def fetch_year_data(year, min_magnitude):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    url = f"{ROOT_URL}?starttime={start_date}&endtime={end_date}&minmagnitude={min_magnitude}&orderby=time"

    print(f"Fetching data of year - {year} ...")
    print(f"Data Source URL: {url}")

    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error while fetching data: {e}")
        if os.path.exists(f"data/year_{year}.csv"):
            print(f"Loading from local cache for year {year}...")
            df = pd.read_csv(f"data/year_{year}.csv")
            return df
        else:
            print(f"No local cache found for year {year}. Skipping.")
        return pd.DataFrame()

    print(f"Successfully fetched {len(df)} rows.")
    return df


def load_historical_data(start_year, end_year, save_directory="data", min_magnitude=4):
    os.makedirs(save_directory, exist_ok=True)

    all_dataframes = []
    for year in range(start_year, end_year + 1):
        df = fetch_year_data(year, min_magnitude)
        if not df.empty:
            df.to_csv(f"{save_directory}/year_{year}.csv", index=False)
        all_dataframes.append(df)

    if not all_dataframes:
        print("No yearly data was loaded.")
        return pd.DataFrame()

    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df.to_csv(f"{save_directory}/historical.csv", index=False)

    print(f"\nTotal records: {len(final_df)}")
    print(f"Saved to: '{save_directory}/historical.csv'")
    return final_df


def process_historical_data():
    if not os.path.exists("data/historical.csv"):
        print("data/historical.csv was not found.")
        return pd.DataFrame()

    df = pd.read_csv("data/historical.csv")

    if df.empty:
        print("historical.csv is empty.")
        df.to_csv("data/historical_processed.csv", index=False)
        return df

    df = df.copy()
    # Let pandas infer datetime format; avoid invalid 'format' argument
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag", "depth"])

    df = df[["id", "time", "latitude", "longitude", "depth", "mag", "place"]]
    df["region"] = df["place"].str.split(",").str[-1].str.strip()  # extract region

    # Reverse geocoding
    coordinates = list(zip(df["latitude"], df["longitude"]))
    results = rg.search(coordinates) if coordinates else []

    # ISO country codes: ensure resulting list matches dataframe length
    if results and len(results) == len(coordinates):
        country_iso = [res.get("cc", "") for res in results]
    else:
        country_iso = [""] * len(df)

    df["country_iso"] = country_iso

    # Convert ISO codes to country names
    countries = []
    for code in df["country_iso"]:
        if code:
            country_obj = pycountry.countries.get(alpha_2=code)
            countries.append(country_obj.name if country_obj else "Unknown")
        else:
            countries.append("Unknown")

    df["country"] = countries

    df.to_csv("data/historical_processed.csv", index=False)
    print(f"Processed records saved: {len(df)}")
    return df


if __name__ == "__main__":
    load_historical_data(2012, 2026)
    process_historical_data()
    # ...
