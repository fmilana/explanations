import random
import pandas as pd


# sample 6 sentences for study introduction (so we pick 3)
def _sample_intro_sentences(probas_df, class_name, q1_score_positive, q3_score_negative):
    class_names = [class_name for class_name in probas_df.columns[7:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]
    # sample first and third sentences (2 each)
    true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 1)]
    tp_q1_examples_df = true_positive_df.loc[(true_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(4).index]

    first_intro_samples = []
    second_intro_samples = []
    third_intro_samples = []

    # sample first sentences
    for i in range(2):
        first_intro_samples.append({
            'sample_id': f'first_samples_{i}',
            'original_sentence': tp_q1_examples_df['original_sentence'].values[i],
            'cleaned_sentence': tp_q1_examples_df['cleaned_sentence'].values[i],
            **{f'proba {class_name}': tp_q1_examples_df[f'proba {class_name}'].values[i] for class_name in class_names}
        })

    # sample third sentences
    for i in range(2, 4):
        third_intro_samples.append({
            'sample_id': f'third_samples_{i}',
            'original_sentence': tp_q1_examples_df['original_sentence'].values[i],
            'cleaned_sentence': tp_q1_examples_df['cleaned_sentence'].values[i],
            **{f'proba {class_name}': tp_q1_examples_df[f'proba {class_name}'].values[i] for class_name in class_names}
        })

    probas_df = probas_df.drop(tp_q1_examples_df.index)

    # sample second sentence
    false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
    pred_columns = [col for col in false_negative_df.columns if col.startswith('pred ')]
    # create a boolean mask that is true for rows where exactly one 'pred' column is 1
    mask = (false_negative_df[pred_columns].sum(axis=1) == 1)
    sub_df = false_negative_df[mask]
    second_intro_sample_df = sub_df.loc[(sub_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(2).index]
    
    # sample second sentences
    for i in range(2):
        second_intro_samples.append({
            'sample_id': f'second_samples_{i}',
            'original_sentence': second_intro_sample_df['original_sentence'].values[i],
            'cleaned_sentence': second_intro_sample_df['cleaned_sentence'].values[i],
            **{f'proba {class_name}': second_intro_sample_df[f'proba {class_name}'].values[i] for class_name in class_names}
        })

    probas_df = probas_df.drop(second_intro_sample_df.index)

    intro_samples = first_intro_samples + second_intro_samples + third_intro_samples

    return intro_samples, probas_df


def _generate_samples_csvs(probas_df, scores_df):
    # remove sentences from probas_df with less than 5 words (when cleaned)
    probas_df = probas_df[probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) >= 5)]
    probas_df = probas_df.reset_index(drop=True)
    
    scores_df.set_index(scores_df.columns[0], inplace=True)

    class_names = [class_name for class_name in probas_df.columns[7:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]

    intro_samples = []

    samples_dict = {}

    for i, class_name in enumerate(class_names):
        q1_score_positive = scores_df.loc['Q1 positive', class_name]
        q3_score_negative = scores_df.loc['Q3 negative', class_name]

        # sample sentences for study intro
        if i == 0:
            intro_samples, probas_df = _sample_intro_sentences(probas_df, class_name, q1_score_positive, q3_score_negative)

        # queries

        # sample query from top 10 positive (at random among top 10 positive)
        top_positive_query_df = probas_df.nlargest(10, f'proba {class_name}').sample(n=1)
        top_positive_query_tuple = (top_positive_query_df['original_sentence'].values[0], top_positive_query_df['cleaned_sentence'].values[0], top_positive_query_df[f'proba {class_name}'].values[0])
        # remove sampled query from probas_df
        probas_df = probas_df.drop(top_positive_query_df.index)

        # sample query from around q1 positive (closest to q1 positive)
        q1_positive_query_df = probas_df.loc[(probas_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(1).index]
        q1_positive_query_tuple = (q1_positive_query_df['original_sentence'].values[0], q1_positive_query_df['cleaned_sentence'].values[0], q1_positive_query_df[f'proba {class_name}'].values[0])
        # remove sampled query from probas_df
        probas_df = probas_df.drop(q1_positive_query_df.index)

        # sample query from around q3 negative (closest to q3 negative)
        q3_negative_query_df = probas_df.loc[(probas_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(1).index]
        q3_negative_query_tuple = (q3_negative_query_df['original_sentence'].values[0], q3_negative_query_df['cleaned_sentence'].values[0], q3_negative_query_df[f'proba {class_name}'].values[0])
        # remove sampled query from probas_df
        probas_df = probas_df.drop(q3_negative_query_df.index)

        # sample query from bottom 10 negative (at random among bottom 10 negative)
        bottom_negative_query_df = probas_df.nsmallest(10, f'proba {class_name}').sample(n=1)
        bottom_negative_query_tuple = (bottom_negative_query_df['original_sentence'].values[0], bottom_negative_query_df['cleaned_sentence'].values[0], bottom_negative_query_df[f'proba {class_name}'].values[0])
        # remove sampled query from probas_df
        probas_df = probas_df.drop(bottom_negative_query_df.index)

        # examples

        # get true positive dfs
        true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 1)]
        # sample 4 examples from top true positives (at random among top 10 true positives)
        tp_top_examples_df = true_positive_df.nlargest(10, f'proba {class_name}').sample(n=4)
        # sample 4 examples from around q1 positives (closest to q1 positive)
        tp_q1_examples_df = true_positive_df.loc[(true_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(4).index]
        # remove sampled queries from probas_df
        probas_df = probas_df.drop(tp_top_examples_df.index)

        # get false positive df
        false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
        # sample 2 examples from top false positives (at random among top 10 false positives)
        fp_top_examples_df = false_positive_df.nlargest(10, f'proba {class_name}').sample(n=2)
        # sample 2 examples from around q1 false positives (closest to q1 positive)
        fp_q1_examples_df = false_positive_df.loc[(false_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(2).index]
        # remove sampled queries from probas_df
        probas_df = probas_df.drop(fp_top_examples_df.index)

        # get false negative df
        false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
        # sample 2 examples from around q3 false negatives (closest to q3 negative)
        fn_q3_examples_df = false_negative_df.loc[(false_negative_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(2).index]
        # sample 2 examples from bottom false negatives
        fn_bottom_examples_df = false_negative_df.nsmallest(10, f'proba {class_name}').sample(n=2)
        # remove sampled queries from probas_df
        probas_df = probas_df.drop(fn_q3_examples_df.index)
        
        # create list of tuples (sentences, proba)
        tp_examples_tuples = list(zip(tp_top_examples_df['original_sentence'], tp_top_examples_df['cleaned_sentence'], tp_top_examples_df[f'proba {class_name}'])) + list(zip(tp_q1_examples_df['original_sentence'], tp_q1_examples_df['cleaned_sentence'], tp_q1_examples_df[f'proba {class_name}']))
        fp_examples_tuples = list(zip(fp_top_examples_df['original_sentence'], fp_top_examples_df['cleaned_sentence'], fp_top_examples_df[f'proba {class_name}'])) + list(zip(fp_q1_examples_df['original_sentence'], fp_q1_examples_df['cleaned_sentence'], fp_q1_examples_df[f'proba {class_name}']))
        fn_examples_tuples = list(zip(fn_q3_examples_df['original_sentence'], fn_q3_examples_df['cleaned_sentence'], fn_q3_examples_df[f'proba {class_name}'])) + list(zip(fn_bottom_examples_df['original_sentence'], fn_bottom_examples_df['cleaned_sentence'], fn_bottom_examples_df[f'proba {class_name}']))
        
        # shuffle lists
        random.shuffle(tp_examples_tuples)
        random.shuffle(fp_examples_tuples)
        random.shuffle(fn_examples_tuples)

        samples_dict[class_name] = {
            'TP Examples Tuples': tp_examples_tuples,
            'FP Examples Tuples': fp_examples_tuples,
            'FN Examples Tuples': fn_examples_tuples,
            'Top Positive Query Tuple': top_positive_query_tuple,
            'Q1 Positive Query Tuple': q1_positive_query_tuple,
            'Q3 Negative Query Tuple': q3_negative_query_tuple,
            'Bottom Negative Query Tuple': bottom_negative_query_tuple
        }

    # save intro samples to csv    
    intro_df = pd.DataFrame(intro_samples)
    intro_df.to_csv('results/intro_samples.csv', index=False)
    
    # save study samples to csv
    samples_df = pd.DataFrame(samples_dict)
    samples_df.to_csv('results/samples.csv')


if __name__ == '__main__':
    try:
        probas_df = pd.read_csv('results/probas.csv')
        scores_df = pd.read_csv('results/scores.csv')
        print('Sampling sentences...')
        _generate_samples_csvs(probas_df, scores_df)
        print('Sentences sampled and stored in results/intro_samples.csv and results/samples.csv.')
    except FileNotFoundError as e:
        print('results/probas.csv and/or results/scores.csv not found. Please run train.py first')
