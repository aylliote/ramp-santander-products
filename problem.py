# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedKFold
from rampwf.score_types.base import BaseScoreType

import warnings
warnings.filterwarnings('ignore')

problem_title = 'Santander Product Recommandations'

_target_column_name = 'producto'
_prediction_label_names = list(np.arange(22))

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

def apk(actual, predicted, k):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


class MeanAveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 0.5

    def __init__(self, name='map@7',precision=7):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        true = np.argmax(y_true, axis = 1).reshape(-1, 1)
        preds = np.flip(np.argsort(y_pred, axis = 1), axis = 1)
        return mapk(true, preds, k = self.precision)

score_types = [MeanAveragePrecision()]

def get_cv(X, y):
    skf = StratifiedKFold(n_splits=5, random_state = 42)
    for train_index, test_index in skf.split(X, y):
        yield train_index, test_index
    

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory = False)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:20000], y_array[:20000]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)

