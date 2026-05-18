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
    except Exception as E:
        print(f"Error while fetching data: {E}")
        return pd.DataFrame()

    print(f"Successfully fetched {len(df)} rows.")

    return df


def load_historical_data(start_year, end_year, save_directory="data", min_magnitude=4):
    all_dataframes = []
    for year in range(start_year, end_year + 1):
        df = fetch_year_data(year, min_magnitude)
        df.to_csv(f"{save_directory}/year_{year}.csv", index=False)
        all_dataframes.append(df)

    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df.to_csv(f"{save_directory}/historical.csv", index=False)

    print(f"\nTotal records: {len(final_df)}")
    print(f"Saved to: '{save_directory}/historical.csv'")


def process_historical_data():
    df = pd.read_csv("data/historical.csv")

    df["time"] = pd.to_datetime(df["time"])

    df = df[["id", "time", "latitude", "longitude", "depth", "mag", "place"]]
    df["region"] = df["place"].str.split(",").str[-1].str.strip()  # extract region

    # Reverse geocoding
    coordinates = list(zip(df["latitude"], df["longitude"]))
    results = rg.search(coordinates)


if __name__ == "__main__":
    load_historical_data(2020, 2026)
