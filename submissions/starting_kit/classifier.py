from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self, C = 1.):
        self.clf = LogisticRegression(C = C)

    def fit(self, X, y):
        self.clf.fit(X, y)
     
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
