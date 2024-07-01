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
from pipeline_tokenizer import Sentence2Tokens
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


def _create_json_entry(sentence, proba, tokenizer, lime_weights, shap_weights, occlusion_weights, interpret_weights):
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

    for word_or_symbol in words_and_symbols:
        if re.fullmatch(r'\w+', word_or_symbol):
            adjusted_lime_weights.append(lime_weights[lime_index])
            adjusted_shap_weights.append(shap_weights[shap_index])
            adjusted_occlusion_weights.append(occlusion_weights[occlusion_index])
            lime_index += 1
            shap_index += 1
            occlusion_index += 1
        else:
            adjusted_lime_weights.append(0.0)
            adjusted_shap_weights.append(0.0)
            adjusted_occlusion_weights.append(0.0)

    for i, word_or_symbol in enumerate(words_and_symbols):
        parts.append({
            'word': word_or_symbol,
            'lime_weight': adjusted_lime_weights[i],
            'shap_weight': adjusted_shap_weights[i],
            'occlusion_weight': adjusted_occlusion_weights[i]
        })

    # create interpret_parts row

    if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
        print('sentence:', sentence)
        print(f'{len(interpret_weights)} interpret_weights: {interpret_weights}')

    interpret_parts = []

    interpret_index = 1 # exclude [CLS]

    while interpret_index < len(interpret_weights) - 1: # exclude [SEP]
        if not words_and_symbols:
            break

        token, weight = interpret_weights[interpret_index]

        cleaned_token = re.sub(r'^##', '', token)

        word_or_symbol = words_and_symbols[0]

        if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
            print(f'checking cleaned_token "{cleaned_token}" against word_or_symbol.lower() "{word_or_symbol.lower()}"')

        if word_or_symbol.isspace():
            if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
                print(f'word_or_symbol "{word_or_symbol}" is a space')
            interpret_parts.append({
                'cleaned_token': ' ',
                'interpret_weight': 0.0
            })
            words_and_symbols.pop(0)
        elif cleaned_token in word_or_symbol.lower():
            if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
                print(f'word_or_symbol "{word_or_symbol}" is not a space and cleaned_token "{cleaned_token}" is in it')
            if cleaned_token == word_or_symbol.lower():
                if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
                    print(f'cleaned_token "{cleaned_token}" is the same as word_or_symbol "{word_or_symbol.lower()}"')
                interpret_parts.append({
                    'cleaned_token': word_or_symbol,
                    'interpret_weight': weight
                })
                words_and_symbols.pop(0)
            else:
                if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
                    print(f'cleaned_token "{cleaned_token}" is not the same as word_or_symbol "{word_or_symbol.lower()}"')
                # only get the part of the word that matches the token
                match = re.search(cleaned_token, word_or_symbol.lower())

                interpret_parts.append({
                    'cleaned_token': word_or_symbol[:match.end()],
                    'interpret_weight': weight
                })

                words_and_symbols[0] = word_or_symbol[match.end():]
            
            interpret_index += 1 # move to the next token only when a match is found
        else:
            if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
                print(f'word_or_symbol "{word_or_symbol}" is not a space and cleaned_token "{cleaned_token}" is not in it')
            words_and_symbols.pop(0)

        if words_and_symbols and not words_and_symbols[0]: # if the word is completely used up
            words_and_symbols.pop(0)

    # interpret_weights = interpret_weights[1: -1] # exclude [CLS] and [SEP]
    # words_and_symbols = tokenizer.tokenize(sentence)
    
    # words_and_symbols = words_and_symbols

    # interpret_parts = []
    # interpret_index = 0

    # if sentence == 'For £10.50 you can get mapo tofu with chicken wings.':
    #     print(f'{len(words_and_symbols)} words_and_symbols: {words_and_symbols}')
    #     print(f'{len(interpret_weights)} interpret_weights: {interpret_weights}')

    # while interpret_index < len(interpret_weights):
    #     if not words_and_symbols:
    #         break

    #     token, weight = interpret_weights[interpret_index]
    #     cleaned_token = re.sub(r'^##', '', token)

    #     word_or_symbol = words_and_symbols[0]

    #     if word_or_symbol.isspace():
    #         interpret_parts.append({
    #             'cleaned_token': ' ',
    #             'interpret_weight': 0.0
    #         })
    #         words_and_symbols.pop(0)



    #     interpret_index += 1


    return {
        'sentence': sentence,
        'classification_score': proba,
        'parts': parts,
        'interpret_parts': interpret_parts
    }


def _generate_file(classifier, samples_df, json_path, lime_optimized):
    samples_dict = _load_samples_as_dict(samples_df)

    json_dict = {}

    pipeline = CustomPipeline(steps=[('tokenizer', Sentence2Tokens(classifier.tokenizer)), ('classifier', classifier)])

    labels = list(samples_dict.keys())

    total_number_of_sentences = sum([len(samples_dict[label]['TP Examples Tuples']) + len(samples_dict[label]['FP Examples Tuples']) + len(samples_dict[label]['FN Examples Tuples']) + 4 for label in labels])
    progress_counter = 0

    example_titles = ['True Positives', 'False Positives', 'False Negatives']
    query_titles = ['Top Positive', 'Q1 Positive', 'Q3 Negative', 'Bottom Negative']

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
                interpret_weights = classifier.get_interpret_weights(sentence, label)

                key = f'{label} {example_titles[i]} {j}'
                json_dict[key] = _create_json_entry(sentence, proba, classifier.tokenizer, lime_weights, shap_weights, occlusion_weights, interpret_weights)

                print(f'{progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
                progress_counter += 1

        tuples_list = [top_positive_query_tuple, q1_positive_query_tuple, q3_negative_query_tuple, bottom_negative_query_tuple]
        zipped_list = list(zip(query_titles, tuples_list))
        # shuffle queries
        random.shuffle(zipped_list)

        for (query_title, (sentence, proba)) in zipped_list:
            lime_weights, shap_weights, occlusion_weights = _get_agnostic_weights(pipeline, labels, sentence, label, proba, lime_optimized)
            interpret_weights = classifier.get_interpret_weights(sentence, label)
            
            key = f'{label} {query_title} Query'
            json_dict[key] = _create_json_entry(sentence, proba, classifier.tokenizer, lime_weights, shap_weights, occlusion_weights, interpret_weights)

            print(f'{progress_counter+1}/{total_number_of_sentences} sentences processed.', end='\r')
            progress_counter += 1

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    try:
        probas_df = pd.read_csv('results/probas.csv')
        scores_df = pd.read_csv('results/scores.csv')
        sampled_df = pd.read_csv('results/samples.csv', index_col=0)
        classifier = MultiLabelProbClassifier()
        print('Generating JSON...')
        _generate_file(classifier, sampled_df, 'results/json/results.json', lime_optimized=False)
        print('\nJSON generated.')
    except FileNotFoundError as e:
        print('Model and/or data not found. Please run train.py and sample.py first.')



# TODO:
# occlusion weights are probably not being calculated correctly (how are they all red?)