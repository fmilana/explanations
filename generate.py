import re
import numpy as np
import pandas as pd
from run_lime import generate_lime
from run_shap import generate_shap
from pathlib import Path
from sklearn.pipeline import make_pipeline
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec


def generate_lime_values(pipeline, categories, sentence):
    lime_dict = generate_lime(pipeline, categories, sentence)

    target = lime_dict["targets"][categories.index(predicted_category)]
    positive_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["pos"]]
    negative_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["neg"]]

    lime_tuples = positive_weight_tuples + negative_weight_tuples
    lime_tuples = sorted(lime_tuples, key=lambda x: int(re.search(r'\[(\d+)\]', x[0]).group(1)) if re.search(r'\[(\d+)\]', x[0]) is not None else -1)

    lime_values = [tuple[1] for tuple in lime_tuples]

    lime_bias = lime_values.pop(0)

    return lime_bias, lime_values


def generate_shap_values(pipeline, categories, sentence):
    shap_array = generate_shap(pipeline, categories, sentence)
    shap_values = shap_array[:, categories.index(predicted_category)].tolist()

    return shap_values


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

prediction = pipeline.predict([sentence]).flatten()
try:
    predicted_category = categories[np.where(prediction==1)[0][0]]
except IndexError:
    predicted_category = "None"
predict_proba = pipeline.predict_proba([sentence]).flatten()

print(f"predicted_category: \"{predicted_category}\"")
print(f"predict_proba: {predict_proba}")

lime_bias, lime_values = generate_lime_values(pipeline, categories, sentence)
shap_values = generate_shap_values(pipeline, categories, sentence)

print(f"lime_bias: {lime_bias}")
print(f"lime_values: {lime_values}")
print(f"shap_values: {shap_values}")