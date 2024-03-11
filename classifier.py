import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForSequenceClassification
    

class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained('models/final').to(device)


    def fit(self, X, Y):
        pass

    
    def predict_proba(self, X):
        inputs = {name: tensor.to(self.model.device) for name, tensor in X.items()}
        output = self.model(**inputs).logits.cpu().detach().numpy()
        probas = torch.sigmoid(torch.from_numpy(output)).numpy()

        ret_list = []

        for list_of_probs in probas:
            sums_to = np.sum(list_of_probs)
            epsilon = 1e-7
            new_probas = np.divide(list_of_probs, sums_to + epsilon)
            ret_list.append(new_probas)
        
        return np.asarray(ret_list)

    
    def predict(self, X):
        return self.predict_proba(X)


    # def predict_proba(self, X):
    #     if len(X) == 1:
    #         self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)[0]
    #         sums_to = sum(self.probas_)
    #         new_probas = [x / sums_to for x in self.probas_] # make probabilities sum to 1 for lime
    #         return new_probas # return list of probas
    #     else:
    #         self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)
    #         ret_list = []
    #         for list_of_probs in self.probas_:
    #             sums_to = sum(list_of_probs)
    #             new_probas = [x / sums_to for x in list_of_probs] # make probabilities sum to 1 for lime
    #             ret_list.append(np.asarray(new_probas))
    #         ret_list = np.asarray(ret_list)
    #         return ret_list # return list of list of probas