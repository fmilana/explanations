import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier


# from https://github.com/TeamHG-Memex/eli5/issues/337
class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):
    number_of_chains = 10
    chains = []

    def __init__(self):
        xgb = XGBClassifier()
        self.chains = [ClassifierChain(xgb, order='random', random_state=i) for i in range(self.number_of_chains)]

    # fit the XGBoost chains
    def fit(self, X, Y):
        for chain in self.chains:
            chain.fit(X, Y)

    # get predictions from the XGBoost chains
    def predict(self, X):
        predictions = np.rint(np.array([chain.predict(X) for chain in self.chains]).mean(axis=0)).astype(int) # predict with the XGBoost chains
        return predictions
    
    # get prediction probas from the XGBoost chains
    def predict_proba(self, X): 
        if len(X) == 1:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)[0]
            sums_to = sum(self.probas_)
            new_probs = [x / sums_to for x in self.probas_]
            new_probs = np.asarray(new_probs).reshape(-1, 1) # reshape to two-dimensional array (for run_counter.py)
            return new_probs
        else:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                # print(sums_to)
                new_probs = [x / sums_to for x in list_of_probs]
                ret_list.append(np.asarray(new_probs))
            ret_list = np.asarray(ret_list)
            return ret_list