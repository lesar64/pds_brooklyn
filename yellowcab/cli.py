from yellowcab.model import fare_amount

import os
import click
import traceback
import pandas as pd
from . import model
from . import io


@click.command()
@click.option('--transform/--no-transform', default=False)
@click.option('-i', default=None)
@click.option('-o', default=None)
@click.option('--predict/--no-predict', default=False)
def main(transform, i, o, predict):

    # überpfügen ob i und o ein pfad sind
    input = None
    output = None
    prediction = pd.DataFrame()
    print(os.path.isdir(os.path.join(o)))
    if (o is not None) and (not os.path.isdir(os.path.join(o))):
        raise NotADirectoryError(str(o) + " is not a directory")
    if i is None:
        raise TypeError("i is None")
    elif not os.path.isfile(i):
        raise FileNotFoundError(str(i) + " is not a file")

    if i is not None:
        try:
            print("File read ...")
            input = io.read_file(path=i)

        except FileNotFoundError as e:
            print(traceback.format_exc(e))

            return

        print("Transforming data ...")
        output = io.create_clean_trip_dataset(data=input, feat=True, ret=True, borough=None)

    if transform:
        if i is None or o is None:
            print("Insufficient file specification, use -i for input path and -o for output path")

            return
        else:
            output.to_parquet(os.path.join(o, "new_data.parquet")) 
            print("File was saved to " + os.path.join(o, "new_data.parquet"))

    elif predict:
        print("Predicting stuff...")

        # print(output.columns)

        # loading all models and predicting if the model is not none
        trip_distance_model = model.trip_distance_model()

        if trip_distance_model is not None: 
            print("Trip distance model loaded")

            prediction["p_trip_distance"] = trip_distance_model.predict(output[
                ['passenger_count', 'PULocationID', 'DOLocationID',
                 'start_month', 'start_day', 'start_hour', 'start_week', 'weekend', 'weekday',
                 'start_location_long', 'start_location_lat', 'end_location_long', 'end_location_lat',
                 'DOBorough', 'DOservice_zone', 'PUBorough', 'PUservice_zone']])

        fare_amount_model = model.fare_amount_lasso_regression()

        if fare_amount_model is not None: 
            print("Fare amount model loaded")

            prediction["p_fare_amout"] = fare_amount_model.predict(output[["trip_distance"]])

        payment_type_model = model.payment_type_model()

        if payment_type_model:
            print("Payment type model loaded")

            prediction["p_payment_type_model"] = payment_type_model.predict(output[
                ['tip_amount', 'congestion_surcharge', 'DOLocationID', 
                'PULocationID', 'total_amount', 'Temperature', 'lockdown']])

        print(prediction.to_numpy())


if __name__ == '__main__':
    main()

