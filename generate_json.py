import random
import re
import json
import xgboost
import joblib
import regex
import pandas as pd
from classifier import MultiLabelProbClassifier
from lime_explain import get_lime_weights
from occlusion_explain import get_occlusion_weights
from shap_explain import get_shap_weights
from vectorizer import Sentence2Vec
from custom_pipeline import CustomPipeline
from tqdm import tqdm


def _create_json_entry(sentence, cleaned_sentence, proba, distance, lime_weights, shap_weights, occlusion_weights):
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
        'score': proba,
        'distance': distance,
        'parts': parts
    }


def _get_all_weights(pipeline, class_names, sentence, class_name, proba, lime_optimized):
    lime_bias, lime_weights = get_lime_weights(pipeline, class_names, sentence, class_name, lime_optimized=lime_optimized)
    shap_weights = get_shap_weights(pipeline, class_names, sentence, class_name)
    occlusion_weights = get_occlusion_weights(pipeline, class_names, sentence, class_name, proba)
    return lime_weights, shap_weights, occlusion_weights


def _generate_file(clf, class_names, samples_df, json_path, lime_optimized):
    sample_dict = samples_df.to_dict(orient='records')

    json_dict = {}

    pipeline = CustomPipeline(steps=[('vectorizer', Sentence2Vec()), ('classifier', clf)])

    for sample in tqdm(sample_dict, desc='Processing sentences'):
        name = sample['name']
        original_sentence = sample['original_sentence']
        cleaned_sentence = sample['cleaned_sentence']
        proba = sample['proba']
        distance = sample['distance']
        class_name = sample['class_name']

        lime_weights, shap_weights, occlusion_weights = _get_all_weights(pipeline, class_names, cleaned_sentence, class_name, proba, lime_optimized)
        json_dict[name] = _create_json_entry(original_sentence, cleaned_sentence, proba, distance, lime_weights, shap_weights, occlusion_weights)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    try:
        clf = xgboost.Booster()
        clf.load_model('model/xgb_model.json')
        clf = MultiLabelProbClassifier(clf) # wrap the model in a MultiLabelProbClassifier for LIME
        train_df = pd.read_csv('data/train.csv')
        class_names = train_df.columns[5:].tolist()
        samples_df = pd.read_csv('results/samples.csv')
        print('Generating JSON...')
        _generate_file(clf, class_names, samples_df, 'output/json/output.json', lime_optimized=True)
        print('\nJSON generated.')
    except FileNotFoundError as e:
        print('Model and/or data not found. Please run train.py and sample.py first.')