import json
import re
import pandas as pd


def _save_weights(class_weights_dict, class_name):
    tp_lime_weights = class_weights_dict['TP']
    fn_lime_weights = class_weights_dict['FN']
    fp_lime_weights = class_weights_dict['FP']

    tp_lime_weights = sorted(tp_lime_weights, key=lambda x: x['lime_weight'], reverse=True)
    fn_lime_weights = sorted(fn_lime_weights, key=lambda x: x['lime_weight'], reverse=True)
    fp_lime_weights = sorted(fp_lime_weights, key=lambda x: x['lime_weight'], reverse=True)

    tp_df = pd.DataFrame(tp_lime_weights)
    fn_df = pd.DataFrame(fn_lime_weights)
    fp_df = pd.DataFrame(fp_lime_weights)

    tp_df.to_csv(f'output/weights/{class_name}_TP_weights.csv', index=False)
    fn_df.to_csv(f'output/weights/{class_name}_FN_weights.csv', index=False)
    fp_df.to_csv(f'output/weights/{class_name}_FP_weights.csv', index=False)


def _list_weights(json_dict):
    weights_dict = {}

    for sample in json_dict.values():
        class_name = sample['class_name']
        category = sample['category']
        parts = sample['parts']

        for part in parts:
            token = part['token']
            ignored = part['ignored']
            lime_weight = part['lime_weight']

            if ignored == 0.0:
                if class_name not in weights_dict.keys():
                    weights_dict[class_name] = {'TP': [], 'FN': [], 'FP': []}
                weights_dict[class_name][category].append({'token': token, 'lime_weight': lime_weight})

    for class_name in weights_dict:
        _save_weights(weights_dict[class_name], class_name)


if __name__ == '__main__':
    try:
        with open('output/json/output.json', 'r') as f:
            json_dict = json.load(f)
    except FileNotFoundError as e:
        print(e)
        print('output/json/output.json not found. Please run generate_json.py first.')
        exit()

    _list_weights(json_dict)

    print('Weights listed and saved in output/weights/.')