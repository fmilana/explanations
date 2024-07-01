from transformers import AutoModel, AutoTokenizer


class Sentence2Embedding:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('models/final')

    
    def fit(self, X, Y):
        pass


    def transform(self, X, y=None):
        return self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
