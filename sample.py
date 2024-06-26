import re
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
        filtered_df = test_probas_df[(test_probas_df[class_name] == 1) & (test_probas_df[f'pred {class_name}'] == 1)]
        num_samples = 3
    elif sample_type == 'FN':
        filtered_df = test_probas_df[(test_probas_df[class_name] == 1) & (test_probas_df[f'pred {class_name}'] == 0)]
        num_samples = 2
    elif sample_type == 'FP':
        filtered_df = test_probas_df[(test_probas_df[class_name] == 0) & (test_probas_df[f'pred {class_name}'] == 1)]
        num_samples = 2
    
    task_df = filtered_df.loc[(filtered_df[f'proba {class_name}'] - midpoint).abs().nsmallest(num_samples).index]
    task_df.reset_index(drop=True, inplace=True)

    for index, row in task_df.iterrows():
        task_tuple = (row['original_sentence'], row['cleaned_sentence'], row['sentence_embedding'], row[f'proba {class_name}'], 0.0)
        task_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple
        all_samples_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple


def _sample_examples(train_probas_df, class_name, task_name, task_tuple, task_dict, all_samples_dict):
    sample_types = {'TP': {'num_task_samples': 3, 'num_example_samples': 6},
                    'FN': {'num_task_samples': 2, 'num_example_samples': 3},
                    'FP': {'num_task_samples': 2, 'num_example_samples': 3}}

    for sample_type, settings in sample_types.items():
        num_task_samples = settings['num_task_samples']
        num_example_samples = settings['num_example_samples']

        if sample_type == 'TP':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'pred {class_name}'] == 1)]
        elif sample_type == 'FN':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'pred {class_name}'] == 0)]
        elif sample_type == 'FP':
            filtered_df = train_probas_df[(train_probas_df[class_name] == 0) & (train_probas_df[f'pred {class_name}'] == 1)]

        for index in range(0, num_task_samples):
            task_tuple_key = f'{class_name} {sample_type} Task {index + 1}'
            task_tuple = task_dict[task_tuple_key]
            filtered_df_with_distance = _add_distance_to_train_probas_df(task_tuple, filtered_df)

            max_example_samples = min(num_example_samples, len(filtered_df_with_distance.nsmallest(num_example_samples*2, 'distance')))
            examples_df = filtered_df_with_distance.nsmallest(num_example_samples*2, 'distance').sample(n=max_example_samples)

            examples_df.reset_index(drop=True, inplace=True)

            for example_index, example_row in examples_df.iterrows():
                example_key = f'{task_name} {sample_type} Example {example_index + 1}'
                example_tuple = (example_row['original_sentence'], example_row['cleaned_sentence'], example_row[f'proba {class_name}'], example_row['distance'])
                all_samples_dict[example_key] = example_tuple


def _generate_samples_csvs(test_probas_df, train_probas_df, scores_df):
    # remove sentences from test_probas_df with 1 or less words when cleaned
    test_probas_df = test_probas_df[test_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 1)]
    test_probas_df = test_probas_df.reset_index(drop=True)
    
    scores_df.set_index(scores_df.columns[0], inplace=True)

    class_names = [class_name for class_name in test_probas_df.columns[7:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]

    all_samples_dict = {}
    task_dict = {}

    for class_name in class_names:
        # get scores for class
        upper_midpoint = scores_df.loc['upper midpoint', class_name]
        lower_midpoint = scores_df.loc['lower midpoint', class_name]

        # task sentences

        # sample 3 task sentences from true positives (closest to upper midpoint)
        _sample_tasks(test_probas_df, class_name, 'TP', upper_midpoint, task_dict, all_samples_dict)
        # sample 2 task sentences from false negatives (closest to lower midpoint)
        _sample_tasks(test_probas_df, class_name, 'FN', lower_midpoint, task_dict, all_samples_dict)
        # sample 2 task sentences from false positives (closest to upper midpoint)
        _sample_tasks(test_probas_df, class_name, 'FP', upper_midpoint, task_dict, all_samples_dict)

        # example sentences

        for task_name, task_tuple in task_dict.items():
            _sample_examples(train_probas_df, class_name, task_name, task_tuple, task_dict, all_samples_dict)

        task_dict = {}

        # remove distance column from task_tuples in all_samples_dict

        pattern = rf'^{re.escape(class_name)} (TP|FN|FP) Task (\d)$'
        for key in all_samples_dict.keys():
            if re.match(pattern, key):
                original_sentence, cleaned_sentence, _, proba, distance = all_samples_dict[key]
                all_samples_dict[key] = (original_sentence, cleaned_sentence, proba, distance)

    # save study samples to csv
    samples_df = pd.DataFrame.from_dict(all_samples_dict, orient='index', columns=['original_sentence', 'cleaned_sentence', 'proba', 'distance'])

    samples_df.to_csv('results/samples.csv')
    print('Samples saved to results/samples.csv')


if __name__ == '__main__':
    try:
        test_probas_df = pd.read_csv('results/test_probas.csv')
        test_probas_df.loc[:, 'sentence_embedding'] = test_probas_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )

        train_probas_df = pd.read_csv('results/train_probas.csv')
        train_probas_df.loc[:, 'sentence_embedding'] = train_probas_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )

        scores_df = pd.read_csv('results/scores.csv')
        print('Sampling sentences...')
        _generate_samples_csvs(test_probas_df, train_probas_df, scores_df)
    except FileNotFoundError as e:
        print('results/test_probas.csv, results/train_probas.csv and/or results/scores.csv not found. Please run train.py first')
