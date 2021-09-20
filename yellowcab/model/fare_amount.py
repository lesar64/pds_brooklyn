from yellowcab.model.model_utils import *
from yellowcab.io import save_model, read_model
import traceback

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


def fare_amount_lasso_regression(x=None, y=None, parameters=None, train=False, load=True, save=False):
    
    # return the loaded model
    if load:
        try:
            return read_model('fare_amount_lasso_regression')
        except FileNotFoundError:
            print("File not found, did you save the model?")
        except Exception as e:
            # everything else, possibly fatal
            print(traceback.format_exc(e))
            return
    
    # train and return the model with the given data and parameters
    if train:
        # train model here
        lasso_model = Pipeline(steps=[('scaler', StandardScaler()),
                                      (('lasso'), Lasso(alpha=0.03))])

        lasso_model.fit(x, y)
            
        # save the model if instructed
        if save:
            save_model('fare_amount_lasso_regression', lasso_model)
            
        # return the trained model
        return lasso_model
    
    
    