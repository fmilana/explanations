from train import train_and_validate
from sample import sample_sentences
from generate import generate_html, generate_json


clf = train_and_validate()

sentence_dict = sample_sentences()

json_dict = generate_html(clf, sentence_dict, "results/html/")

generate_json(json_dict, "results/json/")