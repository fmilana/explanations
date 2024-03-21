from transformers import AutoModel, AutoTokenizer


class Sentence2Tokens:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    
    def fit(self, X, Y):
        pass


    def transform(self, X, y=None):
        return self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
