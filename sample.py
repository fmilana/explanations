import re
import json
import random
import numpy as np
import pandas as pd


def _add_distance_to_train_probas_df(task_tuple, train_probas_df):
    train_probas_df_copy = train_probas_df.copy()
    task_embedding = task_tuple[2]
    train_probas_df_copy.loc[:, 'distance'] = pd.to_numeric(train_probas_df_copy['sentence_embedding'].apply(lambda x: np.linalg.norm(x - task_embedding)))
    
    return train_probas_df_copy


def _sample_tasks(test_probas_df, class_name, sample_type, midpoint, task_dict, all_samples_dict):
    if sample_type == 'TP':
        filtered_df = test_probas_df[(test_probas_df[class_name] == 1) & (test_probas_df[f'proba {class_name}'] >= 0.5)]
        num_samples = 3
    elif sample_type == 'FN':
        filtered_df = test_probas_df[(test_probas_df[class_name] == 1) & (test_probas_df[f'proba {class_name}'] < 0.5)]
        num_samples = 2
    elif sample_type == 'FP':
        filtered_df = test_probas_df[(test_probas_df[class_name] == 0) & (test_probas_df[f'proba {class_name}'] >= 0.5)]
        num_samples = 2
    
    task_df = filtered_df.loc[(filtered_df[f'proba {class_name}'] - midpoint).abs().nsmallest(num_samples).index]

    # drop selected tasks from test_probas_df to avoid duplicates
    indices_to_drop = task_df.index.intersection(test_probas_df.index)
    test_probas_df = test_probas_df.drop(indices_to_drop)

    task_df.reset_index(drop=True, inplace=True)

    for index, row in task_df.iterrows():
        task_tuple = (row['original_sentence'], row['cleaned_sentence'], row['sentence_embedding'], class_name, row[f'proba {class_name}'], 0.0)
        task_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple
        all_samples_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple

    # return test_probas_df with dropped tasks
    return test_probas_df



def _sample_examples(train_probas_df, class_name, task_name, task_tuple, task_dict, all_samples_dict):
    sample_types = {'TP': {'num_task_samples': 3, 'num_example_samples': 6}, 
                    'FN': {'num_task_samples': 2, 'num_example_samples': 9}, # sample more than 3 examples to account for mistakes in coding
                    'FP': {'num_task_samples': 2, 'num_example_samples': 9}} # sample more than 3 examples to account for mistakes in coding

    for sample_type, settings in sample_types.items():
        num_task_samples = settings['num_task_samples']
        num_example_samples = settings['num_example_samples']

        if sample_type == 'TP':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'proba {class_name}'] >= 0.5)]
        elif sample_type == 'FN':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'proba {class_name}'] < 0.5)]
        elif sample_type == 'FP':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 0) & (train_probas_df[f'proba {class_name}'] >= 0.5)]

        for index in range(0, num_task_samples):
            task_tuple_key = f'{class_name} {sample_type} Task {index + 1}'
            task_tuple = task_dict[task_tuple_key]
            filtered_df_with_distance = _add_distance_to_train_probas_df(task_tuple, filtered_df)

            # sort by distance and select examples one by one
            sorted_df = filtered_df_with_distance.sort_values('distance')
            
            examples_list = []

            for _, row in sorted_df.iterrows():
                if len(examples_list) >= num_example_samples:
                    break
                if row['original_sentence'] not in [example['original_sentence'] for example in examples_list]:
                    examples_list.append(row.to_dict())

            examples_df = pd.DataFrame(examples_list)

            # max_example_samples = min(num_example_samples, len(filtered_df_with_distance.nsmallest(num_example_samples*2, 'distance')))
            # examples_df = filtered_df_with_distance.nsmallest(num_example_samples*2, 'distance').sample(n=max_example_samples)

            # drop selected examples from train_probas_df to avoid duplicates
            indices_to_drop = examples_df.index.intersection(train_probas_df.index)
            train_probas_df = train_probas_df.drop(indices_to_drop)

            examples_df.reset_index(drop=True, inplace=True)

            for example_index, example_row in examples_df.iterrows():
                example_key = f'{task_name} {sample_type} Example {example_index + 1}'
                example_tuple = (example_row['original_sentence'], example_row['cleaned_sentence'], class_name, example_row[f'proba {class_name}'], example_row['distance'])
                all_samples_dict[example_key] = example_tuple

    # return train_probas_df with dropped examples
    return train_probas_df


def _generate_samples_csvs(test_probas_df, train_probas_df):
    # remove sentences from test_probas_df with 1 or less words when cleaned
    test_probas_df = test_probas_df[test_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 1)]
    test_probas_df = test_probas_df.reset_index(drop=True)

    class_names = [class_name for class_name in test_probas_df.columns[7:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]

    upper_midpoint = 0.75
    lower_midpoint = 0.25

    all_samples_dict = {}
    task_dict = {}

    for class_name in class_names:
        # task sentences

        # sample 3 task sentences from true positives (closest to upper midpoint)
        test_probas_df = _sample_tasks(test_probas_df, class_name, 'TP', upper_midpoint, task_dict, all_samples_dict)
        # sample 2 task sentences from false negatives (closest to lower midpoint)
        test_probas_df = _sample_tasks(test_probas_df, class_name, 'FN', lower_midpoint, task_dict, all_samples_dict)
        # sample 2 task sentences from false positives (closest to upper midpoint)
        _ = _sample_tasks(test_probas_df, class_name, 'FP', upper_midpoint, task_dict, all_samples_dict)

        # example sentences

        for task_name, task_tuple in task_dict.items():
            train_probas_df = _sample_examples(train_probas_df, class_name, task_name, task_tuple, task_dict, all_samples_dict)

        task_dict = {}

        # remove distance column from task_tuples in all_samples_dict
        pattern = rf'^{re.escape(class_name)} (TP|FN|FP) Task (\d)$'
        for key in all_samples_dict.keys():
            if re.match(pattern, key):
                original_sentence, cleaned_sentence, _, class_name, proba, distance = all_samples_dict[key]
                all_samples_dict[key] = (original_sentence, cleaned_sentence, class_name, proba, distance)

    # save study samples to csv
    samples_df = pd.DataFrame.from_dict(all_samples_dict, orient='index', columns=['original_sentence', 'cleaned_sentence', 'class_name', 'proba', 'distance'])
    # reset the index to turn it into a column, and add the "name" column
    samples_df.reset_index(inplace=True)
    samples_df.rename(columns={'index': 'name'}, inplace=True)
    samples_df.to_csv('results/samples.csv', index=False)
    print('Samples saved to results/samples.csv')


if __name__ == '__main__':
    try:
        test_probas_df = pd.read_csv('results/test/probas.csv')
        test_probas_df.loc[:, 'sentence_embedding'] = test_probas_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )
        # remove rows with less than 3 words when cleaned (to avoid LIME ZeroDivisionError)
        test_probas_df = test_probas_df[test_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 2)]

        train_probas_df = pd.read_csv('results/train/probas.csv')
        train_probas_df.loc[:, 'sentence_embedding'] = train_probas_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )
        # remove rows with less than 3 words when cleaned (to avoid LIME ZeroDivisionError)
        train_probas_df = train_probas_df[train_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 2)]

        print('Sampling sentences...')
        _generate_samples_csvs(test_probas_df, train_probas_df)
    except FileNotFoundError as e:
        print('results/test/probas.csv, results/train/probas.csv and/or results/scores.csv not found. Please run train.py first')
