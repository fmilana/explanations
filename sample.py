import argparse
import re
import json
import random
import numpy as np
import pandas as pd


def _add_distance_to_train_probas_df(task_tuple, train_probas_df):
    train_probas_df_copy = train_probas_df.copy()
    task_embedding = task_tuple[3]
    train_probas_df_copy.loc[:, 'distance'] = pd.to_numeric(train_probas_df_copy['sentence_embedding'].apply(lambda x: np.linalg.norm(x - task_embedding)))
    
    return train_probas_df_copy


def _sample_tasks(test_probas_df, class_name, sample_type, midpoint, task_dict, all_samples_dict, used_sentences):
    sampled = 0

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

    task_df.reset_index(drop=True, inplace=True)

    for index, row in task_df.iterrows():
        task_tuple = (sample_type, row['original_sentence'], row['cleaned_sentence'], row['sentence_embedding'], class_name, 'task', row[f'proba {class_name}'], 0.0)
        task_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple
        all_samples_dict[f'{class_name} {sample_type} Task {index + 1}'] = task_tuple
        used_sentences.add(row['original_sentence'])

        sampled += 1

    if sampled < num_samples:
        print(f'Warning: Not enough {sample_type} Task samples for class "{class_name}" in test set. Expected {num_samples}, got {sampled}')

    return used_sentences


def _sample_examples(train_probas_df, class_name, task_name, task_tuple, sample_type, all_samples_dict, used_sentences):
    sampled = 0

    if sample_type == 'TP':
        filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'proba {class_name}'] >= 0.5)]
        num_samples = 6
    elif sample_type == 'FN':
        filtered_df = train_probas_df[(train_probas_df[class_name] == 1) & (train_probas_df[f'proba {class_name}'] < 0.5)]
        num_samples = 3
    elif sample_type == 'FP':
        filtered_df = train_probas_df[(train_probas_df[class_name] == 0) & (train_probas_df[f'proba {class_name}'] >= 0.5)]
        num_samples = 3

    filtered_df_with_distance = _add_distance_to_train_probas_df(task_tuple, filtered_df)
    sorted_df = filtered_df_with_distance.sort_values('distance')
    examples_list = []

    for _, row in sorted_df.iterrows():
        if len(examples_list) >= num_samples:
            break
        if row['original_sentence'] not in used_sentences:
            examples_list.append(row.to_dict())
            used_sentences.add(row['original_sentence'])

    examples_df = pd.DataFrame(examples_list)
    examples_df.reset_index(drop=True, inplace=True)

    for example_index, example_row in examples_df.iterrows():
        example_key = f'{task_name} {sample_type} Example {example_index + 1}'
        example_tuple = (sample_type, example_row['original_sentence'], example_row['cleaned_sentence'], class_name, 'example', example_row[f'proba {class_name}'], example_row['distance'])
        all_samples_dict[example_key] = example_tuple

        sampled += 1

    if sampled < num_samples:
        print(f'Warning: Not enough {sample_type} Example samples for class "{class_name}" in training set. Expected {num_samples}, got {sampled}')

    return used_sentences


def _generate_samples_csvs(test_probas_df, train_probas_df, class_names):
    # remove sentences from test_probas_df with 1 or less words when cleaned
    test_probas_df = test_probas_df[test_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 1)]
    test_probas_df = test_probas_df.reset_index(drop=True)

    upper_midpoint = 0.75
    lower_midpoint = 0.25

    all_samples_dict = {}
    task_dict = {}

    used_sentences = set() # move this inside the next loop if duplicate samples across classes are allowed

    for class_name in class_names:
        # for each class, sample tasks
        # sample 3 task sentences from true positives (closest to upper midpoint)
        used_sentences = _sample_tasks(test_probas_df, class_name, 'TP', upper_midpoint, task_dict, all_samples_dict, used_sentences)
        # sample 8 task sentences from false negatives (closest to lower midpoint)
        used_sentences = _sample_tasks(test_probas_df, class_name, 'FN', lower_midpoint, task_dict, all_samples_dict, used_sentences)
        # sample 8 task sentences from false positives (closest to upper midpoint)
        used_sentences = _sample_tasks(test_probas_df, class_name, 'FP', upper_midpoint, task_dict, all_samples_dict, used_sentences)

        # for each task, sample examples
        for task_name, task_tuple in task_dict.items():
            # sample 6 examples sentences from true positives (closest to task sentence)
            used_sentences = _sample_examples(train_probas_df, class_name, task_name, task_tuple, 'TP', all_samples_dict, used_sentences)
            # sample 3 examples sentences from false negatives (closest to task sentence)
            used_sentences = _sample_examples(train_probas_df, class_name, task_name, task_tuple, 'FN', all_samples_dict, used_sentences)
            # sample 3 examples sentences from false positives (closest to task sentence)
            used_sentences = _sample_examples(train_probas_df, class_name, task_name, task_tuple, 'FP', all_samples_dict, used_sentences)

        task_dict = {}

        # remove distance column from task_tuples in all_samples_dict
        pattern = rf'^{re.escape(class_name)} (TP|FN|FP) Task (\d)$'
        for key in all_samples_dict.keys():
            if re.match(pattern, key):
                category, original_sentence, cleaned_sentence, _, class_name, sample_type, proba, distance = all_samples_dict[key]
                all_samples_dict[key] = (category, original_sentence, cleaned_sentence, class_name, sample_type, proba, distance)

    # save study samples to csv
    samples_df = pd.DataFrame.from_dict(all_samples_dict, orient='index', columns=['category', 'original_sentence', 'cleaned_sentence', 'class_name', 'sample_type', 'proba', 'distance'])
    # reset the index to turn it into a column, and add the "name" column
    samples_df.reset_index(inplace=True)
    samples_df.rename(columns={'index': 'name'}, inplace=True)
    samples_df.to_csv('results/samples.csv', index=False)
    print(f'Samples saved to results/samples.csv. Total samples: {samples_df.shape[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process class names.')
    parser.add_argument('--class-names', nargs='*', help='Class names to sample tasks and examples for', default=[])
    args = parser.parse_args()

    args_class_names = args.class_names

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
        # remove rows with 2 or fewer cleaned words (to avoid LIME ZeroDivisionError)
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
        # remove rows with 2 or fewer cleaned word (to avoid LIME ZeroDivisionError)
        train_probas_df = train_probas_df[train_probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 2)]

        all_class_names = [class_name for class_name in test_probas_df.columns[5:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]

        try:
            if len(args_class_names) == 0:
                class_names = all_class_names
            else:
                # check if args_class_names are valid
                for class_name in args_class_names:
                    if class_name not in all_class_names:
                        raise ValueError(f'"{class_name}" is not a valid class name. Valid class names are: {all_class_names}')
                class_names = args_class_names
        except ValueError as e:
            print(e)
            print('Please provide valid class names to sample tasks and examples for.')
            print('Valid class names are:', all_class_names)
            exit()

        print('Sampling sentences...')
        _generate_samples_csvs(test_probas_df, train_probas_df, class_names)
    except FileNotFoundError as e:
        print('results/test/probas.csv, results/train/probas.csv and/or results/scores.csv not found. Please run train.py first')
