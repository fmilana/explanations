import re
import sys
import json
import xgboost
import regex
import pandas as pd
from classifier import MultiLabelProbClassifier
from lime_explain import get_lime_weights
from occlusion_explain import get_occlusion_weights
from shap_explain import get_shap_weights
from vectorizer import Sentence2Vec
from custom_pipeline import CustomPipeline
from tqdm import tqdm


def _create_json_entry(sentence, cleaned_sentence, class_name, sample_type, category, proba, distance, lime_weights, shap_weights, occlusion_weights):
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
            parts.append({'token': token, 'ignored': 0.0, 'lime_weight': lime_weights[index], 'shap_weight': shap_weights[index], 'occlusion_weight': occlusion_weights[index]})
            cleaned_words.pop(index)
            lime_weights.pop(index)
            shap_weights.pop(index)
            occlusion_weights.pop(index)
        else:
            parts.append({'token': token, 'ignored': 1.0, 'lime_weight': 0.0, 'shap_weight': 0.0, 'occlusion_weight': 0.0})

    entry = {
        'sentence': sentence,
        'class_name': class_name,
    }

    if sample_type is not None: # if from sample
        entry['sample_type'] = sample_type

    entry.update({
        'category': category,
        'score': proba,
        'distance': distance,
        'parts': parts
    })

    return entry


def _get_all_weights(pipeline, all_class_names, sentence, class_name, proba, lime_optimized):
    lime_bias, lime_weights = get_lime_weights(pipeline, all_class_names, sentence, class_name, lime_optimized=lime_optimized)
    shap_weights = get_shap_weights(pipeline, all_class_names, sentence, class_name)
    occlusion_weights = get_occlusion_weights(pipeline, all_class_names, sentence, class_name, proba)
    return lime_weights, shap_weights, occlusion_weights


def _generate_file_on_train(pipeline, all_class_names, sampled_class_names, train_probas_df, train_json_path, lime_optimized):
    train_probas_dict = train_probas_df.to_dict(orient='records')

    with open(train_json_path, 'w', encoding='utf-8') as f:
        f.write('{\n')  # Start of JSON object
        first_entry = True

        for class_name in sampled_class_names:
            counter = 1

            for counter, entry in enumerate(tqdm(train_probas_dict, desc=f'Processing training sentences for "{class_name}"', file=sys.stdout, dynamic_ncols=False)):
                original_sentence = entry['original_sentence']
                cleaned_sentence = entry['cleaned_sentence']
                proba = entry[f'proba {class_name}']
                # find category based on actual and predicted class
                if entry[class_name] == 1 and entry[f'pred {class_name}'] == 1:
                    category = 'TP'
                elif entry[class_name] == 1 and entry[f'pred {class_name}'] == 0:
                    category = 'FN'
                elif entry[class_name] == 0 and entry[f'pred {class_name}'] == 1:
                    category = 'FP'
                else:
                    category = 'TN'

                lime_weights, shap_weights, occlusion_weights = _get_all_weights(pipeline, all_class_names, cleaned_sentence, class_name, proba, lime_optimized)
                json_entry = _create_json_entry(original_sentence, cleaned_sentence, class_name, None, category, proba, 0.0, lime_weights, shap_weights, occlusion_weights)

                if not first_entry:
                    f.write(',\n')
                first_entry = False

                name = f'{class_name} {counter}'

                f.write(f'  "{name}": {json.dumps(json_entry, ensure_ascii=False)}')

        f.write('\n}\n')  # End of JSON object


def _generate_file_on_samples(pipeline, train_json_path, all_class_names, samples_df, sample_json_path, lime_optimized):
    # Load json_dict from the specified path
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_json = json.load(f)

    sample_dict = samples_df.to_dict(orient='records')
    sample_json_dict = {}

    for sample in tqdm(sample_dict, desc='Processing sampled sentences'):
        name = sample['name']
        category = sample['category']
        original_sentence = sample['original_sentence']
        cleaned_sentence = sample['cleaned_sentence']
        class_name = sample['class_name']
        sample_type = sample['sample_type']
        proba = sample['proba']
        distance = sample['distance']

        # look for matching entry in json_dict based on sentence and class name
        entry = None
        for key, value in train_json.items():
            if value['sentence'] == original_sentence and value['class_name'] == class_name:
                entry = value
                break

        if entry is not None: # sentence found - it is an example and from the train set
            lime_weights = [part['lime_weight'] for part in entry['parts']]
            shap_weights = [part['shap_weight'] for part in entry['parts']]
            occlusion_weights = [part['occlusion_weight'] for part in entry['parts']]

            json_entry = _create_json_entry(original_sentence, cleaned_sentence, class_name, sample_type, category, proba, distance, lime_weights, shap_weights, occlusion_weights)
        else: # sentence not found - it is a task and from the test set
            lime_weights, shap_weights, occlusion_weights = _get_all_weights(pipeline, all_class_names, cleaned_sentence, class_name, proba, lime_optimized)

            json_entry = _create_json_entry(original_sentence, cleaned_sentence, class_name, sample_type, category, proba, 0.0, lime_weights, shap_weights, occlusion_weights)

        sample_json_dict[name] = json_entry

    with open(sample_json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_json_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    try:
        clf = xgboost.Booster()
        clf.load_model('model/xgb_model.json')
        clf = MultiLabelProbClassifier(clf) # wrap the model in a MultiLabelProbClassifier for LIME
        pipeline = CustomPipeline(steps=[('vectorizer', Sentence2Vec()), ('classifier', clf)])

        train_df = pd.read_csv('data/train.csv')
        all_class_names = train_df.columns[5:].tolist()

        train_probas_df = pd.read_csv('results/train/probas.csv')
        # remove rows with 2 or fewer cleaned words (to avoid LIME ZeroDivisionError)
        train_probas_df = train_probas_df[train_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 2)]

        samples_df = pd.read_csv('results/samples.csv')
        sampled_class_names = samples_df['class_name'].unique().tolist()

        train_json_path = 'output/json/train_output.json'
        samples_json_path = 'output/json/samples_output.json'

        print('Generating JSON on train...')
        _generate_file_on_train(pipeline, all_class_names, sampled_class_names, train_probas_df, train_json_path, lime_optimized=True)
        print(f'\nJSON generated in {train_json_path}')

        print('Generating JSON on sampled...')
        _generate_file_on_samples(pipeline, train_json_path, all_class_names, samples_df, samples_json_path, lime_optimized=True)
        print(f'\nJSON generated in {samples_json_path}')
    except FileNotFoundError as e:
        print('Model and/or data not found. Please run train.py and sample.py first.')
        exit()