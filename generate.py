import re
import json

import joblib
from draw import add_title_to_html, add_to_html
from run_lime import generate_lime
from run_shap import generate_shap
from sklearn.pipeline import make_pipeline
from sample import sample_sentences
from vectorizer import Sentence2Vec


def generate_lime_weights(pipeline, class_names, sentence, predicted_class):
    lime_dict = generate_lime(pipeline, class_names, sentence)

    target = lime_dict["targets"][class_names.index(predicted_class)]
    positive_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["pos"]]
    negative_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["neg"]]

    lime_tuples = positive_weight_tuples + negative_weight_tuples
    lime_tuples = sorted(lime_tuples, key=lambda x: int(re.search(r'\[(\d+)\]', x[0]).group(1)) if re.search(r'\[(\d+)\]', x[0]) is not None else -1)

    lime_weights = [tuple[1] for tuple in lime_tuples]

    lime_bias = lime_weights.pop(0)

    return lime_bias, lime_weights


def generate_shap_weights(pipeline, class_names, sentence, class_name):
    shap_array = generate_shap(pipeline, class_names, sentence)
    shap_weights = shap_array[:, class_names.index(class_name)].tolist()

    return shap_weights


def generate_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights):
    words = re.findall(r"\b\w+\b", sentence)
    cleaned_words = re.findall(r"\b\w+\b", cleaned_sentence)

    # create parts row
    parts = []
    # cleaned_words might contain duplicates
    for word in words:
        if word in cleaned_words:
            index = cleaned_words.index(word)
            parts.append((word, lime_weights[index], shap_weights[index]))
            cleaned_words.pop(index)
            lime_weights.pop(index)
            shap_weights.pop(index)
        else:
            parts.append((word, 0, 0))

    return {
        "classification_score": proba,
        "parts": parts
    }


def generate_html(clf, sentence_dict, html_dir):
    html_path = f"{html_dir}results.html"

    json_dict = {}
    # clear html
    with open(html_path, "w") as f:
        pass

    pipeline = make_pipeline(Sentence2Vec(), clf)
    
    class_names = list(sentence_dict.keys())

    for class_name in sentence_dict.keys():
        top_positive_query_tuple = sentence_dict[class_name][0]
        q1_positive_query_tuple = sentence_dict[class_name][1]
        q3_negative_query_tuple = sentence_dict[class_name][2]
        bottom_negative_query_tuple = sentence_dict[class_name][3]
        tp_examples_tuples = sentence_dict[class_name][4]
        fp_examples_tuples = sentence_dict[class_name][5]
        fn_examples_tuples = sentence_dict[class_name][6]

        titles = ['True Positives', 'False Positives', 'False Negatives']

        for i, list_of_tuples in enumerate([tp_examples_tuples, fp_examples_tuples, fn_examples_tuples]):
            add_title_to_html(f'{class_name} {titles[i]}')

            for j, (sentence, cleaned_sentence, proba) in enumerate(list_of_tuples):
                words = re.findall(r"\b\w+\b", sentence)

                lime_bias, lime_weights = generate_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
                shap_weights = generate_shap_weights(pipeline, class_names, cleaned_sentence, class_name)
                json_dict[f"{class_name} {titles[i]} {j}"] = generate_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights)

                add_to_html(html_path, words, proba, lime_bias, lime_weights, shap_weights)
                
        titles = ['Top Positive', 'Q1 Positive', 'Q3 Negative', 'Bottom Negative']        

        for i, (sentence, cleaned_sentence, proba) in enumerate([top_positive_query_tuple, q1_positive_query_tuple, q3_negative_query_tuple, bottom_negative_query_tuple]):
            words = re.findall(r"\b\w+\b", sentence)
            
            add_title_to_html(f'{class_name} {titles[i]} Query')

            lime_bias, lime_weights = generate_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
            shap_weights = generate_shap_weights(pipeline, class_names, cleaned_sentence, class_name)
            json_dict[f"{class_name} {titles[i]} Query"] = generate_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights)

            add_to_html(html_path, words, proba, lime_bias, lime_weights, shap_weights)

    return json_dict
    

def generate_json(json_dict, json_dir):
    json_path = f"{json_dir}results.json"
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    clf = joblib.load("model/model.sav")
    sentence_dict = sample_sentences()
    json_dict = generate_html(clf, sentence_dict, "results/html/")
    generate_json(json_dict, "results/json/")