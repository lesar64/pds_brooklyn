from yellowcab.model.model_utils import *
from yellowcab.io.input import read_model
from yellowcab.io.output import save_model
from yellowcab.io.utils import add_weather_data, add_lockdown

import pandas as pd
import traceback
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None  # default='warn'


# payment_type data transformation
def payment_type_transformation(df):
    # create data
    try:
        df = add_weather_data(df)
        df = add_lockdown(df)

        X = df[['tip_amount',
                'congestion_surcharge',
                'DOLocationID',
                'PULocationID',
                'total_amount',
                'Temperature',
                'lockdown']]

        y = df[['payment_type']].astype(int)

        y = y.to_numpy().ravel()
    except KeyError:
        print('KeyError. Is the format correct?')
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))
        return
    return X, y


# payment_type model method for the random forest model
def payment_type_model(x=None, y=None, parameters=None, train=False, load=True, save=False):

    # return the loaded model
    if load:
        try:
            return read_model('payment_type_rf_final')
        except FileNotFoundError:
            print("File not found, did you save the model?")
            return
        except Exception as e:
            # everything else, possibly fatal
            print(traceback.format_exc(e))
            return

    # train and return the model with the given data and parameters
    if train:
        # pipeline
        numeric_features = ['tip_amount', 'congestion_surcharge', 'total_amount', 'Temperature', 'lockdown']
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                              ('scaler', RobustScaler())])

        categorical_features = ['DOLocationID', 'PULocationID']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

        # train model with data given by x and y
        rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=123,
                                                                         n_jobs=-1,
                                                                         verbose=1,
                                                                         n_estimators=50,
                                                                         max_depth=100,
                                                                         min_samples_split=2,
                                                                         min_samples_leaf=2
                                                                         ))])
        # train model here
        rf_model.fit(x, y)
        # save the model if instructed
        if save:
            save_model('payment_type_rf_final', rf_model)

        # return the trained model
        return rf_model
    return None
