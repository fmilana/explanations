import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from augment import MLSMOTE, get_minority_samples


# from https://github.com/TeamHG-Memex/eli5/issues/337
class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    chains = []

    def __init__(self):
        clf = XGBClassifier()
        number_of_chains = 10
        self.chains = [ClassifierChain(clf, order="random", random_state=i) for i in range(number_of_chains)]

    def fit(self, X, Y): # fit the XGBoost chains
        X, Y = self.oversample(X, Y) # oversample minority classes
        for i, chain in enumerate(self.chains): # fit each XGBoost chain
            chain.fit(X, Y)
            print(f"{i+1}/{len(self.chains)} chains fit")

    def predict(self, X): # get predictions from the XGBoost chains
        # for i, chain in enumerate(self.chains):
        #     print(f"chain {i+1} predict = {chain.predict(X)}")
        # print(f"np.array([chain.predict(X) for chain in self.chains]).mean(axis=0): {np.array([chain.predict(X) for chain in self.chains]).mean(axis=0)}")
        return np.rint(np.array([chain.predict(X) for chain in self.chains]).mean(axis=0)).astype(int) # predict with the XGBoost chains
    
    def predict_proba(self, X): # get prediction probas from the XGBoost chains
        if len(X) == 1:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)[0]
            sums_to = sum(self.probas_)
            new_probs = [x / sums_to for x in self.probas_]
            new_probs = np.asarray(new_probs).reshape(-1, 1) # reshape to two-dimensional array (for run_counter.py)
            return new_probs
        else:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)
            print(self.probas_)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                # print(sums_to)
                new_probs = [x / sums_to for x in list_of_probs]
                ret_list.append(np.asarray(new_probs))
            ret_list = np.asarray(ret_list)
            return ret_list
    
    def oversample(self, X, Y):
        X_shape_old = X.shape
        class_dist = [y/Y.shape[0] for y in Y.sum(axis=0)]
        print("checking for minority classes in train split...")
        X_sub, Y_sub = get_minority_samples(pd.DataFrame(X), pd.DataFrame(Y))

        # print(f"X_sub: {X_sub}")
        # print(f"Y_sub: {Y_sub}")

        if np.shape(X_sub)[0] > 0: # only oversample training set if minority samples are found
            print("minority classes found.")
            print("oversampling...")
            try:
                X_res, Y_res = MLSMOTE(X_sub, Y_sub, round(X.shape[0]/5))       
                X = np.concatenate((X, X_res.to_numpy())) # append augmented samples
                Y = np.concatenate((Y, Y_res.to_numpy())) # to original dataframes
                print("oversampled.")
                class_dist_os = [y/Y.shape[0] for y in Y.sum(axis=0)]
                print("CLASS DISTRIBUTION:")
                print(f"Before MLSMOTE: {X_shape_old}, {class_dist}")
                print(f"After MLSMOTE: {X.shape}, {class_dist_os}")
            except ValueError:
                print("could not oversample because n_samples < n_neighbors in some classes")
        else:
            print("no minority classes.")
        return X, Y