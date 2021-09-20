import pandas as pd
import geopandas as gpd
import numpy as np
import os
import yellowcab


def get_data_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "..", "data")):
        return os.path.join(os.getcwd(), "..", "data")
    elif os.path.isdir(os.path.join(os.getcwd(), "..", "..", "data")):
        return os.path.join(os.getcwd(), "..", "..", "data")
    else:
        raise FileNotFoundError


def add_datetime_columns(df, start_datetime="tpep_pickup_datetime", end_datetime="tpep_dropoff_datetime"):
    df["start_month"] = df[start_datetime].dt.month
    df["start_day"] = df[start_datetime].dt.day
    df["start_hour"] = df[start_datetime].dt.hour
    df["start_week"] = df[start_datetime].dt.isocalendar().week

    df["end_month"] = df[end_datetime].dt.month
    df["end_day"] = df[end_datetime].dt.day
    df["end_hour"] = df[end_datetime].dt.hour
    df["end_week"] = df[end_datetime].dt.isocalendar().week

    return df


def add_duration(df, start_datetime="tpep_pickup_datetime", end_datetime="tpep_dropoff_datetime"):
    df["duration"] = df[end_datetime] - df[start_datetime]
    df["duration"] = df["duration"].dt.total_seconds()

    return df


def add_weekend(df, datetime="tpep_pickup_datetime"):
    df["weekend"] = (df[datetime].dt.weekday > 4).astype(int) # monday = 0, sunday = 6
    df["weekday"] = df[datetime].dt.weekday + 1

    return df


def add_location(df, start_location="PULocationID", end_location="DOLocationID", nan=False):
    gdf = yellowcab.io.read_all_files('geojson')

    geo_data = pd.DataFrame()
    geo_data["objectid"] = gdf["objectid"].astype(int)
    geo_data["longitude"] = gdf.centroid.x
    geo_data["latitude"] = gdf.centroid.y

    if not nan:
        # total range of location_id is 1 - 265
        # there is no geo data for the locations 264 and 265
        df = df[(df[start_location] < 264) & (df[end_location] < 264)]

        print("Trips with location_id above 263 were dropped, since there is no matching geo data.")

    # row location_id has duplicates 2 x 56 and 3 x 103
    # row objectid is not equal to location_id and seems to be the unique id
    # obejctid used to merge geo_data into the dataframe
    df = pd.merge(df, geo_data, left_on=start_location, right_on="objectid", how="left").rename(
        columns={"longitude": "start_location_long", "latitude": "start_location_lat"}).drop(columns=["objectid"])

    df = pd.merge(df, geo_data, left_on=end_location, right_on="objectid", how="left").rename(
        columns={"longitude": "end_location_long", "latitude": "end_location_lat"}).drop(columns=["objectid"])

    if not nan:
    # Not all ids are in the geojson, rows without geo data are dropped
        df.dropna(inplace=True)

    return df


def add_location_dif(df, drop=False):

    df["long_dif"] = abs(df["start_location_long"] - df["end_location_long"])

    df["lat_dif"] = abs(df["start_location_lat"] - df["end_location_lat"])

    if(drop):
        df = df.drop(columns=['start_location_long', 'start_location_lat', 'end_location_long', 'end_location_lat'])

    return df
    

def add_namings(df, start_location="PULocationID", end_location="DOLocationID"):
    csv = yellowcab.io.read_all_files('csv')

    csv_PU = csv.rename(
        {"LocationID": start_location, "Borough": "PUBorough", 'Zone': 'PUZone',
         'service_zone': 'PUservice_zone'},
        axis=1)
    csv_DO = csv.rename(
        {"LocationID": end_location, "Borough": "DOBorough", 'Zone': 'DOZone',
         'service_zone': 'DOservice_zone'},
        axis=1)

    df_new = pd.merge(df, csv_PU, on=start_location)
    df_new = pd.merge(df_new, csv_DO, on=end_location)

    return df_new


def add_weather_data(df):
    # Import Weater Data
    weather_data = pd.read_excel(os.path.join(get_data_path(), 'input/weather_data/weatherdata_brooklyn_2020.xlsx'))

    # Round Timestamp
    df["tpep_pickup_datetime_rd"] = pd.Series(df["tpep_pickup_datetime"]).dt.round("H")

    # Convert to Datetime
    weather_data["Date time"] = pd.to_datetime(weather_data["Date time"])
    df["tpep_pickup_datetime_rd"] = pd.to_datetime(df["tpep_pickup_datetime_rd"])
    df.rename(columns={"tpep_pickup_datetime_rd": "Date time"}, inplace=True)

    # Merge
    df_new = df.merge(weather_data, on='Date time', how='left')
    #del df_new['Date time']

    return df_new


def add_lockdown(df):
    if 'start_week' not in df:
        raise KeyError('The dataframe is missing the star_week column. /n' +
                       'You might need to perform add_datetime_columns on the dataframe')
    lockdown_dict = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5,
                     11: -2, 12: -3, 13: -3, 14: -3, 15: -3, 16: -3, 17: -3, 18: -3,
                     19: -3, 20: 0, 21: 0, 22: 0, 23: 0, 24: 1, 25: 1, 26: 2, 27: 2,
                     28: 3, 29: 3, 30: 4, 31: 4, 32: 4, 33: 4, 34: 4, 35: 4, 36: 4, 37: 4,
                     38: 4, 39: 4, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: -1,
                     47: -1, 48: -1, 49: -1, 50: -1, 51: -1, 52: -1, 53: -1}

    df['lockdown'] = df['start_week'].apply(lambda x: lockdown_dict[x])
    return df


def filter_borough(df, borough="Brooklyn"):
    taxi_zones = yellowcab.io.read_all_files('csv')
    taxi_zones = taxi_zones.loc[taxi_zones['Borough'] == borough].drop('Borough', axis=1).drop('service_zone', axis=1)
    brooklyn_zones = taxi_zones.index.values
    del taxi_zones
    df = df[(df.PULocationID.isin(brooklyn_zones)) | (df.DOLocationID.isin(brooklyn_zones))]
    return df


def get_stats(df, metric='duration', time='month', stat='mean'):
    valid_metric = set(df.columns)
    # valid_time = {'month', 'weekday', 'hour'}
    if metric not in valid_metric:
        raise ValueError("results: metric must be one of %r." % valid_metric)
    # if time not in valid_time:
    #     raise ValueError("results: time must be one of %r." % valid_time)
    if time == 'weekend':
        df[time] = (df['tpep_pickup_datetime'].dt.weekday > 4).astype(int)
    else:
        df[time] = getattr(df['tpep_pickup_datetime'].dt, time)

    if time == 'weekday':
        df[time] += 1

    df = df[[time, metric]]

    try:
        df = getattr(df.groupby([df[time]]), stat)()
    except AttributeError:
        raise ValueError("stat isn't a pandas.groupby function, try mean or std for example.")

    return df


def join_data(parquet, csv):
    # preprocess to join
    csv_edit = csv.rename({"LocationID": 'location_id', "Borough": "PUBorough"}, axis=1)

    # set start data to filter
    parquet_edit = parquet.rename({"PULocationID": 'location_id'}, axis=1)

    # join datasets to get zone names with trip data
    joined_data = pd.merge(parquet_edit, csv_edit, on='location_id', how='left')
    return joined_data
