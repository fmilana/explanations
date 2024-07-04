import os
import re
import xgboost
import numpy as np
import gensim.downloader
from gensim.models import KeyedVectors
import nltk
from nltk import word_tokenize
from datetime import datetime


class Sentence2Vec:
    # https://github.com/RaRe-Technologies/gensim-data
    model_name = 'glove-twitter-50'
    model_file_path = 'embeddings/glove_twitter_50_embeddings'


    def __init__(self):
        np.seterr(divide='raise')
        start = datetime.now()
        nltk.download('punkt')
        if os.path.exists(self.model_file_path):
            print(f'loading {self.model_name} embeddings from disk...')
            self.model = KeyedVectors.load(self.model_file_path)
        else:
            print(f'downloading glove model ({self.model_name})...')
            self.model = gensim.downloader.load(self.model_name)
            print(f'saving {self.model_name} embeddings to disk...')
            self.model.save(self.model_file_path)
        print(f'done in {datetime.now() - start}')


    def fit(self, X=None, y=None): # comply with scikit-learn transformer requirement (model is already trained)
        return self


    def transform(self, X, y=None):  # comply with scikit-learn transformer requirement
        # Convert the list of sentences into a 2D NumPy array of sentence vectors
        sentence_vectors = np.array([self.get_vector(sentence) for sentence in X])
        # Convert the 2D NumPy array into a DMatrix
        return xgboost.DMatrix(sentence_vectors)


    def get_vector(self, sentence):
        # convert to lowercase, ignore all special characters - keep only
        # alpha-numericals and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
        # get word vectors from model
        word_vectors = [self.model[word] for word in word_tokenize(sentence) if word in self.model]
        # create empty sentence vector
        sentence_vector = np.zeros(self.model.vector_size)
        # sentence vector equals average of word vectors
        if (len(word_vectors) > 0):
            sentence_vector = (np.array([sum(word_vector) for word_vector in zip(*word_vectors)])) / sentence_vector.size
        # normalize sentence vector
        norm = np.linalg.norm(sentence_vector)
        if norm != 0:
            sentence_vector = sentence_vector / norm
        else:
            pass

        return sentence_vector