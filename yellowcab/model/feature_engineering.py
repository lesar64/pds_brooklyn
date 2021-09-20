from .. import io

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder


def correlation_analysis(trip_data, drop_columns, dep_var, figname):
    
    #dropping unnecessary columns
    trip_data_filtered = trip_data.drop(columns = drop_columns)
    
    #Using Pearson Correlation feature selection heatmap for numeric input and numeric output
    fig = plt.figure(figsize=(30,10))
    cor = trip_data_filtered.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    io.save_fig(fig, figname)
    plt.show()
    
    #Correlation with output variable
    cor_target = abs(cor[dep_var])
    
    relevant_features = cor_target[cor_target>0.2]
    print("Relevant features:")
    print(relevant_features)

    
def categorical_features(trip_data, columns, drop=True):

    trip_data = trip_data.reset_index(drop=True)

    enc = OneHotEncoder()
    data_fit = enc.fit_transform(trip_data[columns])
    data_fit = pd.DataFrame(data_fit.todense(), columns=enc.get_feature_names(columns))

    if(drop):
        return pd.concat([trip_data.drop(columns = columns), data_fit], axis=1)
    else:
        return pd.concat([trip_data, data_fit], axis=1)
