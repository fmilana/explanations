import json
import re
import pandas as pd


def _save_weights(class_weights_dict, class_name):
    def _save_weights_to_csv(weights, file_path):
        df = pd.DataFrame(weights)
        df = df.round({'lime_weight': 4})
        df.to_csv(file_path, index=False, header=['word', 'weight'])

    def _calculate_and_save_avg_weights(weights, file_path):
        token_dict = {}
        for entry in weights:
            token = entry['token']
            lime_weight = entry['lime_weight']
            if token not in token_dict:
                token_dict[token] = []
            token_dict[token].append(lime_weight)
        avg_lime_weights = [{'token': token, 'lime_weight': sum(lime_weights) / len(lime_weights)} for token, lime_weights in token_dict.items()]
        sorted_avg_lime_weights = sorted(avg_lime_weights, key=lambda x: x['lime_weight'], reverse=True)
        _save_weights_to_csv(sorted_avg_lime_weights, file_path)

    for category in class_weights_dict:
        lime_weights = class_weights_dict[category]
        sorted_lime_weights = sorted(lime_weights, key=lambda x: x['lime_weight'], reverse=True)
        _save_weights_to_csv(sorted_lime_weights, f'output/weights/{class_name}_{category}_train_weights.csv')
        _calculate_and_save_avg_weights(sorted_lime_weights, f'output/weights/{class_name}_{category}_train_avg_weights.csv')

    all_lime_weights = class_weights_dict['TP'] + class_weights_dict['FN'] + class_weights_dict['FP'] + class_weights_dict['TN']
    sorted_all_lime_weights = sorted(all_lime_weights, key=lambda x: x['lime_weight'], reverse=True)
    _save_weights_to_csv(sorted_all_lime_weights, f'output/weights/{class_name}_all_train_weights.csv')
    _calculate_and_save_avg_weights(sorted_all_lime_weights, f'output/weights/{class_name}_all_train_avg_weights.csv')


def _list_weights(json_dict):
    weights_dict = {}

    for entry in json_dict.values():
        class_name = entry['class_name']
        category = entry['category']
        parts = entry['parts']

        for part in parts:
            token = part['token']
            ignored = part['ignored']
            lime_weight = part['lime_weight']

            if ignored == 0.0:
                if class_name not in weights_dict.keys():
                    weights_dict[class_name] = {'TP': [], 'FN': [], 'FP': [], 'TN': []}
                weights_dict[class_name][category].append({'token': token, 'lime_weight': lime_weight})

    for class_name in weights_dict:
        _save_weights(weights_dict[class_name], class_name)


if __name__ == '__main__':
    try:
        with open('output/json/train_output.json', 'r') as f:
            json_dict = json.load(f)
    except FileNotFoundError as e:
        print(e)
        print('output/json/train_output.json not found. Please run generate_json.py first.')
        exit()

    _list_weights(json_dict)

    print('Weights listed and saved in output/weights/')