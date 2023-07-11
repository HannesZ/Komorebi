import sys

from model import RegressorModel, ClassifierModel

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, log_loss

# lightxgb:
from lightgbm import LGBMRegressor, LGBMClassifier
try: module_name = LGBMRegressor.__module__[:LGBMRegressor.__module__.index(".")]
except: module_name = LGBMRegressor.__module__
print("lightgbm version: " + sys.modules[module_name].__version__)


class LGBMRegressorModel(RegressorModel):
    
    def __init__(self, param_grid=None):
        if param_grid is None:
            self.param_grid = default_LGBMRegressor_param_grid
        else:
            self.param_grid = param_grid

    def get_pipeline(self):
        pipeline = Pipeline([
            ("regressor", LGBMRegressor())
        ])
        return pipeline, self.param_grid

default_LGBMRegressor_param_grid = {
    'regressor__learning_rate': [0.02, 0.1],
    'regressor__n_estimators': [250, 350],
    'regressor__objective': ['regression'],
    'regressor__boosting_type': ['gbdt', 'goss'],
    'regressor__subsample': [0.75],
    'regressor__reg_lambda':[0.05, 0.08],
    'classifier__random_state': [42]
}



class LGBMClassifier(ClassifierModel):

    def __init__(self, param_grid=None):
        if param_grid is None:
            self.param_grid = default_LGBMClassifier_param_grid
        else:
            self.param_grid = param_grid

    def get_pipeline(self):
        pipeline = Pipeline([
            ("classifier", LGBMClassifier())
        ])
        return pipeline, parameters

default_LGBMClassifier_param_grid = {
    'classifier__learning_rate': [0.02, 0.1],
    'classifier__n_estimators': [250, 350],
    'classifier__objective': ['multiclass'],
    'classifier__subsample': [0.75, 0.8],
    'classifier__reg_lambda':[0.05, 0.08],
    'classifier__random_state': [42]
}

