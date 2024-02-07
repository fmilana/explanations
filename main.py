from train import train_and_validate
from sample import get_sentence_dictionary
from generate import generate_html


clf = train_and_validate()

sentence_dict = get_sentence_dictionary()

generate_html(clf, sentence_dict)