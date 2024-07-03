import xgboost
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# from https://github.com/TeamHG-Memex/eli5/issues/337
class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf):
        self.clf = clf


    def fit(self, X, Y):
        self.clf.fit(X, Y)


    def predict_proba_normalized(self, X):
        if X.num_row() == 1:
            probas = self.clf.predict(X)[0]
            sums_to = sum(probas)
            new_probas = [x / sums_to for x in probas] # make probabilities sum to 1 for LIME
            return new_probas # return list of probas
        else:
            list_of_probas = self.clf.predict(X)
            new_list_of_probas = []
            for probas in list_of_probas:
                sums_to = sum(probas)
                new_probas = [x / sums_to for x in probas]
                new_list_of_probas.append(np.asarray(new_probas))
            return np.asarray(new_list_of_probas)

    
    def predict(self, X):
        return self.predict_proba_normalized(X)
    