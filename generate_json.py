import ast
import random
import re
import json
import pandas as pd
import joblib
import regex
from lime_explain import get_lime_weights
from occlusion_explain import get_occlusion_weights
from shap_explain import get_shap_weights
from vectorizer import Sentence2Vec
from custom_pipeline import CustomPipeline


def _load_samples_as_dict(samples_df):
    # define rows that contain lists of tuples
    list_of_tuples_rows = ['TP Examples Dict', 'FP Examples Dict', 'FN Examples Dict']
    # define rows that contain single tuples
    tuples_rows = ['Q1 False Positive Query Tuples', 'Upper Median False Positive Query Tuples', 'Lower Median False Negative Query Tuples', 'Q3 False Negative Query Tuples']
    # convert the strings in the list_of_tuples_rows back into lists of tuples
    for row in list_of_tuples_rows:
        samples_df.loc[row] = samples_df.loc[row].apply(ast.literal_eval)
    # convert the strings in the tuples_rows back into tuples
    for row in tuples_rows:
        samples_df.loc[row] = samples_df.loc[row].apply(ast.literal_eval)
    # convert df to dictionary
    samples_dict = samples_df.to_dict()

    return samples_dict


def _get_all_weights(pipeline, class_names, sentence, class_name, proba, lime_optimized):
    lime_bias, lime_weights = get_lime_weights(pipeline, class_names, sentence, class_name, lime_optimized=lime_optimized)
    shap_weights = get_shap_weights(pipeline, class_names, sentence, class_name)
    occlusion_weights = get_occlusion_weights(pipeline, class_names, sentence, class_name, proba)
    return lime_weights, shap_weights, occlusion_weights


def _create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights, occlusion_weights):
    # extract tokens from sentence (including punctuation, symbols and whitespace)
    tokens = regex.findall(r'\b\p{L}+\b|\S|\s', sentence)
    # extract words from cleaned_sentence
    cleaned_words = re.findall(r'\b\w+\b', cleaned_sentence)
    # create parts row
    parts = []
    # cleaned_words might contain duplicates
    for token in tokens:
        if token in cleaned_words:
            index = cleaned_words.index(token)
            parts.append({'token': token, 'lime_weight': lime_weights[index], 'shap_weight': shap_weights[index], 'occlusion_weight': occlusion_weights[index]})
            cleaned_words.pop(index)
            lime_weights.pop(index)
            shap_weights.pop(index)
            occlusion_weights.pop(index)
        else:
            parts.append({'token': token, 'lime_weight': 0.0, 'shap_weight': 0.0, 'occlusion_weight': 0.0})

    return {
        'sentence': sentence,
        'classification_score': proba,
        'parts': parts
    }


def _generate_file(clf, samples_df, json_path, lime_optimized):
    samples_dict = _load_samples_as_dict(samples_df)

    json_dict = {}

    pipeline = CustomPipeline(steps=[('vectorizer', Sentence2Vec()), ('classifier', clf)])

    class_names = list(samples_dict.keys())

    total_number_of_sentences = sum(
        len(samples_dict[class_name]['TP Examples Dict']['Top']) +
        len(samples_dict[class_name]['TP Examples Dict']['Q1']) +
        len(samples_dict[class_name]['FP Examples Dict']['Top']) +
        len(samples_dict[class_name]['FP Examples Dict']['Q1']) +
        len(samples_dict[class_name]['FN Examples Dict']['Q3']) +
        len(samples_dict[class_name]['FN Examples Dict']['Bottom']) +
        len(samples_dict[class_name]['Q1 False Positive Query Tuples']) +
        len(samples_dict[class_name]['Upper Median False Positive Query Tuples']) +
        len(samples_dict[class_name]['Lower Median False Negative Query Tuples']) +
        len(samples_dict[class_name]['Q3 False Negative Query Tuples'])
        for class_name in class_names
    )
    progress_counter = 0

    for class_name in class_names:
        tp_examples_dict = samples_dict[class_name]['TP Examples Dict']
        fp_examples_dict = samples_dict[class_name]['FP Examples Dict']
        fn_examples_dict = samples_dict[class_name]['FN Examples Dict']
        q1_fp_query_tuples = samples_dict[class_name]['Q1 False Positive Query Tuples']
        upper_median_fp_query_tuples = samples_dict[class_name]['Upper Median False Positive Query Tuples']
        lower_median_fn_query_tuples = samples_dict[class_name]['Lower Median False Negative Query Tuples']
        q3_fn_query_tuples = samples_dict[class_name]['Q3 False Negative Query Tuples']

        titles = ['True Positives', 'False Positives', 'False Negatives']

        examples_dicts = [tp_examples_dict, fp_examples_dict, fn_examples_dict]

        for i, title in enumerate(titles):
            examples_dict = examples_dicts[i]

            for key, list_of_tuples in examples_dict.items():
                for j, (sentence, cleaned_sentence, proba) in enumerate(list_of_tuples):
                    lime_weights, shap_weights, occlusion_weights = _get_all_weights(pipeline, class_names, cleaned_sentence, class_name, proba, lime_optimized)
                    json_dict[f'{class_name} {key} {title} {j+1}'] = _create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights, occlusion_weights)
                    
                    print(f'{progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
                    progress_counter += 1

        titles = ['Q1 False Positive Query', 'Upper Median False Positive Query', 'Lower Median False Negative Query', 'Q3 False Negative Query']
        query_tuples_list = [q1_fp_query_tuples, upper_median_fp_query_tuples, lower_median_fn_query_tuples, q3_fn_query_tuples]

        for title, query_tuples in zip(titles, query_tuples_list):
            random.shuffle(query_tuples)
            for i, (sentence, cleaned_sentence, proba) in enumerate(query_tuples):
                lime_weights, shap_weights, occlusion_weights = _get_all_weights(pipeline, class_names, cleaned_sentence, class_name, proba, lime_optimized)
                json_dict[f'{class_name} {title} {i+1}'] = _create_json_entry(sentence, cleaned_sentence, proba, lime_weights, shap_weights, occlusion_weights)

                print(f'{progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
                progress_counter += 1

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    try:
        clf = joblib.load('model/model.sav')
        sampled_df = pd.read_csv('results/samples.csv', index_col=0)
        print('Generating JSON...')
        _generate_file(clf, sampled_df, 'results/json/results.json', lime_optimized=True)
        print('\nJSON generated.')
    except FileNotFoundError as e:
        print('Model and/or data not found. Please run train.py and sample.py first.')
