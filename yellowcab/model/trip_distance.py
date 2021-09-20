from yellowcab.model.model_utils import *
from yellowcab.io import save_model, read_model, save_csv
import traceback

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


scenario1 = {
    "numerical_features": ['passenger_count', 'start_month', 'start_day', 
        'start_hour', 'start_week', 'start_location_long', 'start_location_lat'],

    "categorical_features": ['PULocationID', 'PUBorough', 'PUservice_zone',
        'weekend', 'weekday'],

    "alpha": 0.01,

    "degree": 3
}

scenario2 = {
    # "numerical_features": ['passenger_count', 'start_month', 'start_day', 
    #     'start_hour', 'start_week', 'long_dif', 'lat_dif'],
    "numerical_features": ['passenger_count', 'start_month', 'start_day', 
        'start_hour', 'start_week', 'start_location_long', 'start_location_lat',
        'end_location_long', 'end_location_lat'],

    "categorical_features": ['PULocationID', 'PUBorough', 'PUservice_zone',
        'DOLocationID', 'DOBorough', 'DOservice_zone', 'weekend', 'weekday']
}


# template for model method
def trip_distance_model(X=None, y=None, scenario=2, train=False, load=True, save=False):

    if (scenario != 1) and (scenario != 2):
        raise ValueError("Parameter scenario has to be one or two!")

    # return the loaded model
    if load:
        try:
            return read_model('s' + str(scenario) + '_trip_distance_mlp')
        except FileNotFoundError:
            print("File not found, did you save the model?")
        except Exception as e:
            # everything else, possibly fatal
            print(traceback.format_exc(e))
            return

    # train and return the model with the given data and parameters
    if train:
        # train model with data given by x and y

        model = None

        if scenario == 1:
            model = scenario_one(X, y)

        if scenario == 2:
            model = scenario_two(X, y)
        # train model here

        # save the model if instructed
        if save & (model != None):
            save_model('s' + scenario + 'trip_distance_mlp', model)

        # return the trained model
        return model


# target location is not known
def scenario_one(X_train, y_train):

    pipe = None

    # create training pipeline with given numerical and categorical features
    try:
        pipe = training_pipe(scenario1["numerical_features"], scenario1["categorical_features"],
            scenario1["degree"], regressor=Lasso(scenario1["alpha"]))
    except KeyError as k:
        print("Input has to contain " + scenario1["numerical_features"] + " and " + scenario1["categorical_features"] + " and the target [\"trip_distance\"]")
        print(traceback.format_exc(k))
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))
        return

    return pipe


# target location is known
# additonal features 'long_dif', 'lat_dif', 'DOLocationID', 'DOBorough' and 'DOservice_zone'
def scenario_two(X_train, y_train):

    pipe = None

    # create training pipeline with given numerical and categorical features
    try:
        pipe = training_pipe(scenario2["numerical_features"], scenario2["categorical_features"],
            1 , regressor=MLPRegressor(verbose=True))
    except KeyError as k:
        print("Input has to contain " + scenario2["numerical_features"] + " and " + scenario2["categorical_features"] + " and the target [\"trip_distance\"]")
        print(traceback.format_exc(k))
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))
        return

    return pipe


def training_pipe(num_feat, cat_feat, degree=None, regressor=Lasso()):

    # two options for gridsearch and designated value
    if degree == None:
        numeric_transformer = Pipeline(steps=[
            ('poly', PolynomialFeatures()),
            ('scaler', RobustScaler())])
    else:
        numeric_transformer = Pipeline(steps=[
            ('poly', PolynomialFeatures(degree)),
            ('scaler', RobustScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # transformation for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feat),
            ('cat', categorical_transformer, cat_feat)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', regressor)])

    return pipe


def gridsearch(scoring, num_feat, cat_feat, poly_degree=[1, 2, 3, 4], alpha=np.linspace(0,0.01,5), regressor=Lasso(), random_state=42):

    pipe = training_pipe(num_feat, cat_feat, regressor=regressor)

    hyperparameter_grid = {
        "preprocessor__num__poly__degree": poly_degree,
        "regressor__alpha": alpha,
    }

    grid = GridSearchCV(pipe, hyperparameter_grid, verbose=2, n_jobs=4, scoring=scoring, cv=KFold(shuffle=True, random_state=random_state))

    return grid


def evaluate_grid(grid, X_train, y_train, save_as=None):

    # fit grid to given data
    grid.fit(X_train, y_train)

    # save result to dataframe
    grid_df = pd.DataFrame(grid.cv_results_["params"])
    grid_df["loss"] = grid.cv_results_["mean_test_score"]

    print(grid.best_params_)

    if save_as != None:
        save_csv(save_as, grid_df)

    return grid_df


# deprecated use gridsearch(..., regressor=Lasso()) instead
def lasso_gridsearch(scoring, poly_degree=[1, 2, 3], alpha=np.linspace(0,1.5,11), random_state=42):

    scaler = RobustScaler()
    poly = PolynomialFeatures()
    lasso = Lasso()

    pipe = Pipeline([
        ("poly", poly),
        ("scaler", scaler),
        ("lasso", lasso)
    ])

    hyperparameter_grid = {
        "poly__degree": poly_degree,
        "lasso__alpha": alpha
    }

    grid = GridSearchCV(pipe, hyperparameter_grid, verbose=True, n_jobs=4, scoring=scoring, cv=KFold(shuffle=True, random_state=random_state))

    return grid

# deprecated use gridsearch(..., regressor=Ridge()) instead
def ridge_gridsearch(scoring, poly_degree=[1, 2, 3], alpha=np.linspace(0,1.5,11), random_state=42):

    scaler = RobustScaler()
    poly = PolynomialFeatures()
    ridge = Ridge()

    pipe = Pipeline([
        ("poly", poly),
        ("scaler", scaler),
        ("ridge", ridge)
    ])

    hyperparameter_grid = {
        "poly__degree": poly_degree,
        "ridge__alpha": alpha
    }

    grid = GridSearchCV(pipe, hyperparameter_grid, verbose=True, n_jobs=4, scoring=scoring, cv=KFold(shuffle=True, random_state=random_state))

    return grid
