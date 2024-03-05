import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# class Sentence2Vec:
#     # https://github.com/RaRe-Technologies/gensim-data
#     model_name = 'glove-twitter-50'
#     model_file_path = 'embeddings/glove_twitter_50_embeddings'

#     vector_sentence_dict = {}

#     def __init__(self):
#         start = datetime.now()
#         nltk.download('punkt')
#         if os.path.exists(self.model_file_path):
#             print(f'loading {self.model_name} embeddings from disk...')
#             self.model = KeyedVectors.load(self.model_file_path)
#         else:
#             print(f'downloading glove model ({self.model_name})...')
#             self.model = gensim.downloader.load(self.model_name)
#             print(f'saving {self.model_name} embeddings to disk...')
#             self.model.save(self.model_file_path)
#         print(f'done in {datetime.now() - start}')

#     def fit(self, X=None, y=None): # comply with scikit-learn transformer requirement (model is already trained)
#         return self

#     def transform(self, X, y=None): # comply with scikit-learn transformer requirement
#         return np.array([self.get_vector(sentence) for sentence in X])

#     def get_vector(self, sentence):
#         # convert to lowercase, ignore all special characters - keep only
#         # alpha-numericals and spaces
#         sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
#         # get word vectors from model
#         word_vectors = [self.model[word] for word in word_tokenize(sentence) if word in self.model]
#         # create empty sentence vector
#         sentence_vector = np.zeros(self.model.vector_size)
#         # sentence vector equals average of word vectors
#         if (len(word_vectors) > 0):
#             sentence_vector = (np.array([sum(word_vector) for word_vector in zip(*word_vectors)])) / sentence_vector.size

#         self.vector_sentence_dict[sentence_vector.tobytes()] = sentence

#         return sentence_vector


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
        print(type(X))
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
