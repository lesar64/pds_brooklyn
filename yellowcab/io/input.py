# from .utils import *
from yellowcab.io.utils import *
import pandas as pd
import os
import pickle
from pathlib import Path
import geopandas as gpd
import datetime as dt


def read_all_files(file_type, raw=False):
    if file_type == "csv":
        taxi_zones_csv = pd.read_csv(os.path.join(get_data_path(), "input/taxi_zones", "taxi_zones.csv"))
        return taxi_zones_csv
    elif file_type == "geojson":
        taxi_zones_geojson = gpd.read_file(os.path.join(get_data_path(), "input/taxi_zones", "NYC Taxi Zones.geojson"))
        return taxi_zones_geojson
    elif file_type == "parquet":
        if raw:
            data_dir = Path(get_data_path(), "input/trip_data/raw")
            trip_data_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in data_dir.glob('*.parquet'))
            trip_data_df.reset_index(inplace=True, drop=True)
            return trip_data_df
        else:
            path = os.path.join(get_data_path(), "input", "trip_data", "clean_data.parquet")
            try:
                clean_trip_data_df = pd.read_parquet(path)
                return clean_trip_data_df
            except FileNotFoundError:
                print("Data file not found. Path was " + path)
                print("Create file with create_clean_trip_dataset()")


# default template
def read_file(path=os.path.join(get_data_path(), "input", "trip_data", "raw", "01.parquet")):
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model(name="model.pkl"):
    model = pickle.load(open(os.path.join(get_data_path(), "output", "models", name + "_model.pkl"), 'rb'))
    return model


def create_clean_trip_dataset(soft_duration_cutoff=13000, borough='Brooklyn', data=None, feat=False, save=False, ret=False):
    
    df = None

    # load data
    if data is None:
        print('Reading the raw data')
        df = read_all_files('parquet', raw=True)
    else:
        print('Using given data')
        df = data

    # filter borough
    if borough is not None:
        print('Filtering the borough')
        df = filter_borough(df, borough)

    # add duration
    print('Adding duration for filtering')
    df = add_duration(df)

    # filter negative and extrem values
    print('Filtering the extreme values')
    df = df[df['tpep_pickup_datetime'].between('2020-01-01 00:00:00', '2020-12-31 23:59:59')]
    df = df[df['tpep_dropoff_datetime'].between('2020-01-01 00:00:00', '2020-12-31 23:59:59')]
    df = df[df['passenger_count'] > 0]
    df = df[df['trip_distance'].between(0.01, 1000)]
    df = df[df['fare_amount'].between(0.01, 7000)]
    df = df[df['tip_amount'] >= 0]
    df = df[df['tolls_amount'] >= 0]
    df = df[df['total_amount'].between(0, 7000)]
    df = df[df['congestion_surcharge'] >= 0]
    df = df[df['duration'].between(1, 57600)]  # cut off at 16h

    midnight = dt.datetime(2020, 1, 1, hour=0, minute=0, second=0)
    df = df[((df['duration'] < soft_duration_cutoff) |
             ((df['tpep_pickup_datetime'].dt.time != midnight.time()) &
             (df['tpep_dropoff_datetime'].dt.time != midnight.time())))]

    df = df.dropna()

    # add columns
    print('Augmenting the raw data')
    df = add_datetime_columns(df)
    df = add_weekend(df)
    df = add_location(df)
    df = add_namings(df)

    # addional features
    if feat:
        df = add_lockdown(df)
        df = add_weather_data(df)
        df = add_location_dif(df)

    print('Resetting index and saving')
    df.reset_index(inplace=True, drop=True)
    if save:
        df.to_parquet(os.path.join(get_data_path(), "input", "trip_data", "clean_data.parquet"))
        print('Done. File is at:' + os.path.join(get_data_path(), "input", "trip_data", "clean_data.parquet"))
    
    if ret:
        return df
    else:
        return None
