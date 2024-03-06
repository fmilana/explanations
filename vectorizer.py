import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch


class Sentence2Embedding:
    model_ckpt = 'bert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    def __init__(self):
        self.model = AutoModel.from_pretrained(self.model_ckpt).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    

    def fit(self, X, Y):
        pass


    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, str):
            return self._get_embeddings([X])
        batch_size = 64
        batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
        embeddings = []
        for batch in batches:
            batch_embeddings = self._get_embeddings(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    
    def _extract_hidden_states(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return last_hidden_state[:, 0].cpu().numpy()
    

    def _get_embeddings(self, batch):
        tokenized_batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        embeddings = self._extract_hidden_states(tokenized_batch)
        return embeddings
