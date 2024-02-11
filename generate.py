import re
import json
import joblib
import regex
from draw import add_title_to_html, add_to_html
from lime_explain import get_lime_weights
from shap_explain import get_shap_weights
from sklearn.pipeline import make_pipeline
from sample import sample_sentences
from vectorizer import Sentence2Vec


def create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights):
    # extract tokens from sentence (including punctuation, symbols and whitespace)
    tokens = regex.findall(r"\b\p{L}+\b|\S|\s", sentence)
    # extract words from cleaned_sentence
    cleaned_words = re.findall(r"\b\w+\b", cleaned_sentence)
    # create parts row
    parts = []
    # cleaned_words might contain duplicates
    for token in tokens:
        if token in cleaned_words:
            index = cleaned_words.index(token)
            parts.append({"token": token, "lime_weight": lime_weights[index], "shap_weight": shap_weights[index]})
            cleaned_words.pop(index)
            lime_weights.pop(index)
            shap_weights.pop(index)
        else:
            parts.append({"token": token, "lime_weight": 0, "shap_weight": 0})

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

    for class_name in class_names:
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

                lime_bias, lime_weights = get_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
                shap_weights = get_shap_weights(pipeline, class_names, cleaned_sentence, class_name)

                json_dict[f"{class_name} {titles[i]} {j}"] = create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights)

                add_to_html(html_path, words, proba, lime_bias, lime_weights, shap_weights)
                
        titles = ['Top Positive', 'Q1 Positive', 'Q3 Negative', 'Bottom Negative']        

        for i, (sentence, cleaned_sentence, proba) in enumerate([top_positive_query_tuple, q1_positive_query_tuple, q3_negative_query_tuple, bottom_negative_query_tuple]):
            words = re.findall(r"\b\w+\b", sentence)
            
            add_title_to_html(f'{class_name} {titles[i]} Query')

            lime_bias, lime_weights = get_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
            shap_weights = get_shap_weights(pipeline, class_names, cleaned_sentence, class_name)

            json_dict[f"{class_name} {titles[i]} Query"] = create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights)

            add_to_html(html_path, words, proba, lime_bias, lime_weights, shap_weights)

    return json_dict
    

def generate_json(json_dict, json_dir):
    json_path = f"{json_dir}results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        clf = joblib.load("model/model.sav")
        sentence_dict = sample_sentences()
        json_dict = generate_html(clf, sentence_dict, "results/html/")
        generate_json(json_dict, "results/json/")
    except FileNotFoundError as e:
        print("Model not found. Please run train.py first.")
