import re
import numpy as np
import pandas as pd
from draw import create_html
from preprocess import remove_stop_words
from run_lime import generate_lime
from run_shap import generate_shap
from pathlib import Path
from sklearn.pipeline import make_pipeline
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec


def generate_lime_weights(pipeline, categories, sentence):
    lime_dict = generate_lime(pipeline, categories, sentence)

    target = lime_dict["targets"][categories.index(predicted_category)]
    positive_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["pos"]]
    negative_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["neg"]]

    lime_tuples = positive_weight_tuples + negative_weight_tuples
    lime_tuples = sorted(lime_tuples, key=lambda x: int(re.search(r'\[(\d+)\]', x[0]).group(1)) if re.search(r'\[(\d+)\]', x[0]) is not None else -1)

    lime_weights = [tuple[1] for tuple in lime_tuples]

    lime_bias = lime_weights.pop(0)

    return lime_bias, lime_weights


def generate_shap_weights(pipeline, categories, sentence):
    shap_array = generate_shap(pipeline, categories, sentence)
    shap_weights = shap_array[:, categories.index(predicted_category)].tolist()

    return shap_weights


train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 2:])

print("calling make_pipeline")
pipeline = make_pipeline(Sentence2Vec(), MultiLabelProbClassifier())
print("done calling make_pipeline")
print("calling pipeline.fit")
pipeline.fit(X_train, Y_train)
print("done calling pipeline.fit")

categories = ["food and drinks", "place", "people", "opinions"]

sentence = "This is not cooking that redefines the very notion of Greek food."
# sentence = "Head chef Graham Chatham, who has cooked at Rules and Daylesford Organic, treats them with old school care, attention and at times, maternal indulgence."
cleaned_sentence = remove_stop_words(sentence)

prediction = pipeline.predict([cleaned_sentence]).flatten()
try:
    predicted_category = categories[np.where(prediction==1)[0][0]]
except IndexError:
    predicted_category = "None"
predict_proba = pipeline.predict_proba([cleaned_sentence]).flatten()

predicted_category_proba = predict_proba[categories.index(predicted_category)]

print(f"predicted_category: \"{predicted_category}\"")
print(f"predict_proba: {predict_proba}")

lime_bias, lime_weights = generate_lime_weights(pipeline, categories, cleaned_sentence)
shap_weights = generate_shap_weights(pipeline, categories, cleaned_sentence)

create_html(
    sentence, 
    predicted_category, 
    predicted_category_proba, 
    lime_bias, 
    lime_weights, 
    shap_weights
    )