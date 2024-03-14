import torch
import ast
import random
import re
import json
import pandas as pd
import regex
from lime_explain import get_lime_weights
from occlusion_explain import get_occlusion_weights
from shap_explain import get_shap_weights
from custom_pipeline import CustomPipeline
from vectorizer import Sentence2Embedding
from classifier import MultiLabelProbClassifier


def _load_samples_as_dict(samples_df):
    # define rows that contain lists of tuples
    list_of_tuples_rows = ['TP Examples Tuples', 'FP Examples Tuples', 'FN Examples Tuples']
    # define rows that contain single tuples
    tuples_rows = ['Top Positive Query Tuple', 'Q1 Positive Query Tuple', 'Q3 Negative Query Tuple', 'Bottom Negative Query Tuple']
    # convert the strings in the list_of_tuples_rows back into lists of tuples
    for row in list_of_tuples_rows:
        samples_df.loc[row] = samples_df.loc[row].apply(ast.literal_eval)
    # convert the strings in the tuples_rows back into tuples
    for row in tuples_rows:
        samples_df.loc[row] = samples_df.loc[row].apply(ast.literal_eval)
    # convert df to dictionary
    samples_dict = samples_df.to_dict()

    return samples_dict


def _get_agnostic_weights(pipeline, labels, sentence, label, proba, lime_optimized):
    lime_bias, lime_weights = get_lime_weights(pipeline, labels, sentence, label, lime_optimized=lime_optimized)
    shap_weights = get_shap_weights(pipeline, labels, sentence, label)
    occlusion_weights = get_occlusion_weights(pipeline, labels, sentence, label, proba)
    return lime_weights, shap_weights, occlusion_weights


def _create_json_entry(sentence, proba, lime_weights, shap_weights, occlusion_weights, interpret_weights):
    # extract words from sentence (including punctuation, symbols and whitespace)
    words_and_symbols = re.findall(r'(\W|\w+)', sentence)

    spaces = [word_or_symbol.isspace() for word_or_symbol in words_and_symbols]
    not_spaces = [word_or_symbol for word_or_symbol in words_and_symbols if not word_or_symbol.isspace()]

    

    # create parts row
    parts = []
    
    # append words and their weights to parts

    adjusted_lime_weights = []
    adjusted_shap_weights = []
    adjusted_occlusion_weights = []

    lime_index = shap_index = occlusion_index = 0

    print(f'sentence = {sentence}')
    print(f'{len(words_and_symbols)} words_and_symbols: {words_and_symbols}')
    print(f'{len(lime_weights)} lime weights: {lime_weights}')

    print(f'len(words_and_symbols): {len(words_and_symbols)}')
    print(f'len(spaces): {len(spaces)}')
    print(f'len(not_spaces): {len(not_spaces)}')

    for word_or_symbol in words_and_symbols:
        if word_or_symbol.isspace():
            adjusted_lime_weights.append(0.0)
            adjusted_shap_weights.append(0.0)
            adjusted_occlusion_weights.append(0.0)
        else:
            adjusted_lime_weights.append(lime_weights[lime_index])
            adjusted_shap_weights.append(shap_weights[shap_index])
            adjusted_occlusion_weights.append(occlusion_weights[occlusion_index])
            lime_index += 1
            shap_index += 1
            occlusion_index += 1

    print(f'len(word_or_symbol): {len(word_or_symbol)}')
    print(f'len(adjusted_lime_weights): {len(adjusted_lime_weights)}')
    print(f'len(adjusted_shap_weights): {len(adjusted_shap_weights)}')
    print(f'len(adjusted_occlusion_weights): {len(adjusted_occlusion_weights)}')

    for i, word_or_symbol in enumerate(word_or_symbol):
        parts.append({
            'word': word_or_symbol,
            'lime_weight': adjusted_lime_weights[i],
            'shap_weight': adjusted_shap_weights[i],
            'occlusion_weight': adjusted_occlusion_weights[i]
        })

    # create interpret_parts row
    interpret_parts = []
    # remove '##' from tokens and add space after each word
    pattern = r'^##'
    for i, (token, weight) in enumerate(interpret_weights[1:-1]): # exclude [CLS] and [SEP]
        if not re.search(pattern, token) and i != 0:
            interpret_parts.append({'cleaned_token': ' ', 'interpret_weight': 0.0})
        cleaned_token = re.sub(pattern, '', token)
        interpret_parts.append({'cleaned_token': cleaned_token, 'interpret_weight': weight})

    return {
        'sentence': sentence,
        'classification_score': proba,
        'parts': parts,
        'interpret_parts': interpret_parts
    }


def _generate_file(classifier, samples_df, json_path, lime_optimized):
    samples_dict = _load_samples_as_dict(samples_df)

    json_dict = {}

    pipeline = CustomPipeline(steps=[('vectorizer', Sentence2Embedding()), ('classifier', classifier)])

    labels = list(samples_dict.keys())

    total_number_of_sentences = sum([len(samples_dict[label]['TP Examples Tuples']) + len(samples_dict[label]['FP Examples Tuples']) + len(samples_dict[label]['FN Examples Tuples']) + 4 for label in labels])
    progress_counter = 0

    titles = ['Top Positive', 'Q1 Positive', 'Q3 Negative', 'Bottom Negative']

    agnostic_weights_dict = {}

    for label in labels:
        tp_examples_tuples = samples_dict[label]['TP Examples Tuples']
        fp_examples_tuples = samples_dict[label]['FP Examples Tuples']
        fn_examples_tuples = samples_dict[label]['FN Examples Tuples']
        top_positive_query_tuple = samples_dict[label]['Top Positive Query Tuple']
        q1_positive_query_tuple = samples_dict[label]['Q1 Positive Query Tuple']
        q3_negative_query_tuple = samples_dict[label]['Q3 Negative Query Tuple']
        bottom_negative_query_tuple = samples_dict[label]['Bottom Negative Query Tuple']

        for i, list_of_tuples in enumerate([tp_examples_tuples, fp_examples_tuples, fn_examples_tuples]):
            for j, (sentence, proba) in enumerate(list_of_tuples):

                lime_weights, shap_weights, occlusion_weights = _get_agnostic_weights(pipeline, labels, sentence, label, proba, lime_optimized)

                agnostic_weights_dict[f'{label} {titles[i]} {j}'] = (sentence, proba, lime_weights, shap_weights, occlusion_weights)

                print(f'1st run: {progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
                progress_counter += 1

        tuples_list = [top_positive_query_tuple, q1_positive_query_tuple, q3_negative_query_tuple, bottom_negative_query_tuple]
        zipped_list = list(zip(titles, tuples_list))
        # shuffle queries
        random.shuffle(zipped_list)

        for (title, (sentence, proba)) in zipped_list:
            lime_weights, shap_weights, occlusion_weights = _get_agnostic_weights(pipeline, labels, sentence, label, proba, lime_optimized)
            agnostic_weights_dict[f'{label} {title} Query'] = (sentence, proba, lime_weights, shap_weights, occlusion_weights)

            print(f'1st run: {progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
            progress_counter += 1

    # free up memory between model agnostic and interpret weight calculations
    torch.cuda.empty_cache()

    progress_counter = 0

    # calculate interpret weights and create json entry
    for key, value in agnostic_weights_dict.items():
        pattern = r'(' + '|'.join(map(re.escape, titles)) + r'|\s\d+|\sQuery)'
        label = re.sub(pattern, '', key).strip()

        interpret_weights = classifier.get_interpret_weights(value[0], label)
        json_dict[key] = _create_json_entry(value[0], value[1], value[2], value[3], value[4], interpret_weights)
        print(f'2nd run: {progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
        progress_counter += 1

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    try:
        probas_df = pd.read_csv('results/probas.csv')
        scores_df = pd.read_csv('results/scores.csv')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = MultiLabelProbClassifier()
        sampled_df = pd.read_csv('results/samples.csv', index_col=0)
        print('Generating JSON...')
        _generate_file(classifier, sampled_df, 'results/json/results.json', lime_optimized=False)
        print('\nJSON generated.')
    except FileNotFoundError as e:
        print('Model and/or data not found. Please run train.py and sample.py first.')



# TODO:
# remove stopwords from transformers-interpret heat maps? or include them in the agnostic heatmaps since bert uses them?
# no need to split agnostic and interpret weight calculations into two separate loops? no difference in cuda mem? 
# ^ if so, revert to previous version (no agnostic dict, no 2nd run, no empty_cache call)
# occlusion weights are probably not being calculated correctly (how are they all red?)