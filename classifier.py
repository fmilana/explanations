import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer
    

class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained('models/final').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('models/final')
        self.cls_explainer = MultiLabelClassificationExplainer(self.model, self.tokenizer)


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

    
    def get_interpret_weights(self, sentence, label):
            if sentence == 'As ever, Korean barbecue brings with it an awful lot of admin.':
                print('sentence:', sentence)
                print('label:', label)
                print(f'word_attributions: {[(token, attribute) for token, attribute in self.cls_explainer(sentence)[label]]}')

            return [(token, attribute) for token, attribute in self.cls_explainer(sentence)[label]]
