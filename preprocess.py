import re
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


download('stopwords')


def get_stop_words():
    stop_words = list(set(stopwords.words('english')))
    extra_stop_words = open('data/extra_stopwords.txt', 'r', encoding='utf-8').read().split(',')

    stop_words += extra_stop_words

    return stop_words


def remove_stop_words(text):
    stop_words = get_stop_words()

    print(f"removing stop words from: {text}")

    word_tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
    text = ' '.join(word_tokens)

    return text
