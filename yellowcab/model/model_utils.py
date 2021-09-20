from .. import io
from .. import model

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import LinearSVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def get_features(data, known, predict, sample=None, random_state=42, cat_feat=["PULocationID"]):

    if sample != None:
        data = data.sample(sample, random_state=random_state)

    data = data[known]

    X = model.categorical_features(data, columns=cat_feat)
    X.drop(columns=predict, inplace=True)
    y = data[predict].to_numpy()
    y = y.ravel()

    return X, y


def test_regression_model(model, X_train, X_test, y_train, y_test):

    y_test_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)

    train =[]

    train.append(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    train.append(metrics.mean_absolute_error(y_train, y_train_pred))

    test = []

    test.append(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    test.append(metrics.mean_absolute_error(y_test, y_test_pred))

    df = pd.DataFrame(data={'Training set': train, 'Test set': test})
    df.rename(index={0: "RSME", 1:"MAE"}, inplace=True)

    return df


def compare_regression(data, known, predict, sample, scoring='r2', random_state=42, cat_feat=["PULocationID", "PUBorough", "PUservice_zone"]):

    X, y = get_features(data, known, predict, sample=sample, random_state=random_state, cat_feat=cat_feat)

    scaler = StandardScaler().fit_transform(X)

    # prepare models
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('L1', Lasso()))
    models.append(('L2', Ridge()))
    # models.append(('SVR', LinearSVR()))
    models.append(('CART', DecisionTreeRegressor()))
    # models.append(('KNN', KNeighborsRegressor()))
    models.append(('NN', MLPRegressor()))

    compare_models(models, X, y, scoring, random_state=random_state)


def compare_classification(data, known, predict, sample, scoring='accuracy', random_state=42, cat_feat=["PULocationID", "PUBorough", "PUservice_zone", "DOLocationID", "DOBorough", "DOservice_zone"]):

    X, y = get_features(data, known, predict, sample=sample, random_state=random_state, cat_feat=cat_feat)

    scaler = StandardScaler().fit_transform(X)

    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('SGD', SGDClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    compare_models(models, X, y, scoring, random_state=random_state)


# prepare models by:

# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('SGD', SGDClassifier()))

def compare_models(models, X, y, scoring, figname=None, random_state=42):
    results = []
    names = []

    # evaluate each model in turn
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=random_state, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    if figname != None:
        io.save_fig(fig, figname)