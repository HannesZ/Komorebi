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

    def get_pipeline(self):
        if self.param_grid is not None:
            parameters = self.param_grid
        else:
            parameters = default_LGBMRegressor_param_grid
        pipeline = Pipeline([
            ("regressor", LGBMRegressor(learning_rate=0.1, n_estimators=250, objective='regression', subsample=0.75, reg_lambda=0.06))
        ])
        return pipeline, parameters

default_LGBMRegressor_param_grid = {
    'regressor__learning_rate': [0.02, 0.1],
    'regressor__n_estimators': [250, 350],
    'regressor__objective': ['regression'],
    'regressor__boosting_type': ['gbdt', 'goss'],
    'regressor__subsample': [0.75],
    'regressor__reg_lambda':[0.05]
}



class LGBMClassifier(ClassifierModel):

    def get_pipeline(self):
        if self.param_grid is not None:
            parameters = self.param_grid
        else:
            parameters = default_LGBMClassifier_param_grid
        pipeline = Pipeline([
            ("classifier", LGBMClassifier(learning_rate=0.1, n_estimators=250, objective='multiclass', subsample=0.75, reg_lambda=0.06))
        ])
        return pipeline, parameters

default_LGBMClassifier_param_grid = {
    'classifier__learning_rate': [0.02, 0.1],
    'classifier__n_estimators': [250, 350],
    'classifier__objective': ['multiclass'],
    'classifier__subsample': [0.75, 0.8],
    'classifier__reg_lambda':[0.05, 0.08]
}

