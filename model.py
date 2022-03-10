import sys
import os
import time
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from copy import copy
#from tkinter.messagebox import NO

# from ml_insights import model_performance, feature_importance
import pandas as pd

# sklearn:
from sklearn.model_selection import train_test_split, GridSearchCV

try: module_name = train_test_split.__module__[:train_test_split.__module__.index(".")]
except: module_name = train_test_split.__module__
print("sklearn version: " + sys.modules[module_name].__version__)

class Model(ABC):
    def __init__(self, data, target, cat_cols, num_cols=None, sample_weights=None, model_name='', param_grid=None, data_source=None):

        self.data = data
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.sample_weights = sample_weights
        self.model_name = model_name
        self.train_cols = self.get_train_cols()
        self.data_train = None
        self.model = None
        self.param_grid = param_grid
        self.data_source = data_source
        # set negative sample_weights to 0
        if sample_weights is not None:
            negative_sample_weights = data[sample_weights] < 0
            data[sample_weights].clip(0, axis=0, inplace=True)
            print(f"Clipping negative weights effecting {negative_sample_weights.shape[0]} data records")

    def fit(self, data_train, tune=False):

        pipeline, param_grid = self.get_pipeline()

         # hyper parameter tuning
        if tune:
            scoring = "neg_log_loss" if isinstance(self, ClassifierModel) else "neg_mean_squared_error"
            self.model = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=4, scoring=scoring)
        else:
            self.model = pipeline

        # train model:
        begin = time.time()

        if self.sample_weights is not None:
            kwargs = {self.model.estimator.steps[-1][0] + '__sample_weight': data_train[self.sample_weights]} #https://github.com/scikit-learn/scikit-learn/issues/18159
            self.model.fit(data_train[self.train_cols], data_train[self.target], **kwargs)
        else:
            self.model.fit(data_train[self.train_cols], data_train[self.target])
        end = time.time()
        print(f"time of training: {end - begin} seconds")

    def train(self, train_rows=None, tune=None, export=True, info_cols=None):
        if train_rows is None:
            data_train, data_test = train_test_split(self.data, test_size=0.01, random_state=1)
        else:
            if train_rows.dtype.name == 'bool':
                assert 0 < train_rows.values.sum() < self.data.shape[0], 'train-data not specified correctly. Needs to be a subset of all rows of data'
                data_train, data_test = self.data[train_rows], self.data[~train_rows]
            elif train_rows.dtype.name[:3] == 'int':
                assert 0 < train_rows.shape[0] < self.data.shape[0], 'train-data not specified correctly. Needs to be a subset of all rows of data'
                data_train, data_test = self.data.loc[train_rows], self.data.loc[~train_rows]
            else:
                raise ValueError('train_rows must be boolean series or index array')

        self.fit(data_train, tune=tune)

        if export:
            self.predict_and_export(data_test, info_cols=info_cols)
        
        return self

    def export_grid_result(self, data_source, n_samples):
        if isinstance(self.model, GridSearchCV):
            test_results = pd.DataFrame(self.model.cv_results_).sort_values(by=['rank_test_score'])
            test_results['data_source'] = data_source
            test_results['n_samples']= n_samples
            now = datetime.now()
            test_result_name = now.strftime(
                './tuning_output/' + str(self.model) + 'tuning-results-sample#size-' + str(n_samples) + '-v%Y-%m-%dT%H-%M-%S.xlsx')
            test_results.to_excel(test_result_name)
            return test_results

    @abstractmethod
    def get_pipeline(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def get_train_cols(self):
        # validate column specification and return list of column names of features for training

        required_columns = set([self.target] + self.cat_cols)
        train_cols = self.cat_cols if self.num_cols is None else self.cat_cols + self.num_cols

        # check target in columns and numerical:
        assert self.target in self.data.columns, f"Target {self.target} column missing in data columns."
        
        # check target-column data-type:
        if self.__class__.__base__.__name__ == 'ClassifierModel':
            assert pd.api.dtypes.is_categorical_dtype(self.data[self.target]), f"column {self.target} needs to be categorical"
        else:
            assert pd.api.types.is_numeric_dtype(self.data[self.target]), f"column {self.target} needs to be numerical"

        # check if cat_cols exist and are categorical:
        assert all(x in self.data.columns for x in self.cat_cols), f"missing cat_cols in data"
        assert all(pd.api.types.is_categorical_dtype(self.data[c]) for c in self.cat_cols), "columns specified in cat_cols need to be categorical"
                
        # check num_cols:
        if self.num_cols is not None:
            assert all(x in self.data.columns for x in self.num_cols), f"missing num_cols in data"
            assert all(pd.api.types.is_numeric_dtype(self.data[c]) for c in self.num_cols), f"columns specified in num_cols need to be numerical"
        
        # check sample-weights:
        if self.sample_weights is not None:
            required_columns = required_columns.union(self.sample_weights)  
            assert self.sample_weights in self.data.columns, f"sample-weights column missing in data"
            assert pd.api.types.is_numeric_dtype(self.data[self.sample_weights]), f"sample-weights need to be numerical"

        return train_cols

    def predict_and_export(self, data, info_cols=None):
        now = datetime.now()
        # save model as pickle-file and predictions as .csv:
        PATH = './output/'
        version = now.strftime('-v%Y-%m-%dT%H-%M-%S')
        # first save model-file by creating shallow copy by removing the attribute data, then pickle:
        self_copy = copy(self)
        self_copy.data = None
        with open(PATH + self.model_name + version + '.pkl', 'wb') as outfile:
            pickle.dump(self_copy, outfile)

        if info_cols is None:
            cols = self.train_cols + [self.target] if self.sample_weights is None else self.train_cols + [self.sample_weights, self.target]
        else:
            info_cols = [col for col in info_cols if col in data.columns] # make sure to include non-missing info-cols only
            cols = self.train_cols + [self.target] + info_cols if self.sample_weights is None else self.train_cols + [self.sample_weights, self.target] + info_cols
        y_predicted = self.predict(data)
        if self.sample_weights is not None:
            y_predicted_weighted = y_predicted.mul(data[self.sample_weights], axis=0)
        res = pd.concat([data[cols], y_predicted], axis=1)
        os.makedirs(PATH, exist_ok=True)
        res.to_excel(PATH + self.model_name + '.xlsx')

    def balance_model(self, data):
        preds = self.predict(data)
        sum_actuals = sum(self.data[self.target])
        neg_vals = preds < 0
        if neg_vals[neg_vals].shape[0] > 0:
            preds_maxed = preds.copy
            preds_maxed.loc[neg_vals] = 0
            sum_predicted_maxed = sum(preds_maxed)
        sum_predicted = sum(self.predict(data).values)[0]
        factor_calib = sum_actuals/sum_predicted
        return factor_calib

    def export(self, path='./output/'):
        now = datetime.now()
        # save model as pickle file
        version = now.strftime('-v%Y-%m-%dT%H-%M-%S')

        # first save model-file by creating shallow copy by removing the attribute data, then pickle:
        self_copy = copy(self)
        self_copy.data = None
        with open(path + self.model_name + version + '.pkl', 'wb') as outfile:
            pickle.dump(self_copy, outfile)


class RegressorModel(Model):

    @abstractmethod
    def get_pipeline(self):
        pass

    def predict(self, data):
        y_predicted = self.model.predict(data[self.train_cols])
        ret = pd.DataFrame(data=y_predicted, index=data.index, columns=['preds'])
        return ret

class ClassifierModel(Model):

    @abstractmethod
    def get_pipeline(self):
        pass

    def predict(self, data):
        columns = self.model.classes_.astype(str)
        y_predicted = self.model.predict_proba(data[self.train_cols])
        ret = pd.DataFrame(data=y_predicted, index =data.index, column=columns)
        return ret
        






