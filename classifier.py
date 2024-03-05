import numpy as np
import GPUtil
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier


# from https://github.com/TeamHG-Memex/eli5/issues/337
class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):
    number_of_chains = 10
    chains = []
    gpu = len(GPUtil.getGPUs()) > 0


    def __init__(self):
        if self.gpu:
            xgb = XGBClassifier(tree_method='hist', device='cuda')
        else:
            xgb = XGBClassifier()
        self.chains = [ClassifierChain(xgb, order='random', random_state=i) for i in range(self.number_of_chains)]


    def fit(self, X, Y):
        for chain in self.chains:
            chain.fit(X, Y)


    def predict_proba(self, X):
        single_instance = len(X) == 1

        if single_instance:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)[0]
            sums_to = sum(self.probas_)
            new_probas = [x / sums_to for x in self.probas_] # make probabilities sum to 1 for lime
            return new_probas # return list of probas
        else:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                new_probas = [x / sums_to for x in list_of_probs] # make probabilities sum to 1 for lime
                ret_list.append(np.asarray(new_probas))
            return np.asarray(ret_list) # return list of list of probas

    
    def predict(self, X):
        return self.predict_proba(X)
    