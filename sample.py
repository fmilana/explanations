import random
import pandas as pd


def _generate_samples_csvs(probas_df, scores_df):
    # remove sentences from probas_df with less than 5 words (when cleaned)
    probas_df = probas_df[probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) >= 5)]
    probas_df = probas_df.reset_index(drop=True)
    
    scores_df.set_index(scores_df.columns[0], inplace=True)

    class_names = [class_name for class_name in probas_df.columns[7:].tolist() if not (class_name.startswith('pred') or class_name.startswith('proba'))]

    samples_dict = {}

    probas_df_copy = probas_df.copy()

    counter = 0

    # keep sampling until we have valid samples (or until 100 tries)
    while counter < 100:
        valid_sampling = True

        for class_name in class_names:
            q1_score_positive = scores_df.loc['Q1 positive', class_name]
            q3_score_negative = scores_df.loc['Q3 negative', class_name]

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
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(tp_top_examples_df.index)
            # get true positive dfs again
            true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 1)]
            # sample 4 examples from around q1 positives (closest to q1 positive)
            tp_q1_examples_df = true_positive_df.loc[(true_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(4).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(tp_q1_examples_df.index)

            # get false positive df
            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
            # sample 2 examples from top false positives (at random among top 10 false positives)
            fp_top_examples_df = false_positive_df.nlargest(10, f'proba {class_name}').sample(n=2)
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fp_top_examples_df.index)
            # get false positive df again
            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
            # sample 2 examples from around q1 false positives (closest to q1 positive)
            fp_q1_examples_df = false_positive_df.loc[(false_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(2).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fp_q1_examples_df.index)

            # get false negative df
            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
            # sample 2 examples from around q3 false negatives (closest to q3 negative)
            fn_q3_examples_df = false_negative_df.loc[(false_negative_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(2).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fn_q3_examples_df.index)
            # get false negative df again
            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
            # sample 2 examples from bottom false negatives
            fn_bottom_examples_df = false_negative_df.nsmallest(10, f'proba {class_name}').sample(n=2)
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fn_bottom_examples_df.index)
            
            # create list of tuples (sentences, proba)
            tp_examples_tuples = list(zip(tp_top_examples_df['original_sentence'], tp_top_examples_df['cleaned_sentence'], tp_top_examples_df[f'proba {class_name}'])) + list(zip(tp_q1_examples_df['original_sentence'], tp_q1_examples_df['cleaned_sentence'], tp_q1_examples_df[f'proba {class_name}']))
            fp_examples_tuples = list(zip(fp_top_examples_df['original_sentence'], fp_top_examples_df['cleaned_sentence'], fp_top_examples_df[f'proba {class_name}'])) + list(zip(fp_q1_examples_df['original_sentence'], fp_q1_examples_df['cleaned_sentence'], fp_q1_examples_df[f'proba {class_name}']))
            fn_examples_tuples = list(zip(fn_q3_examples_df['original_sentence'], fn_q3_examples_df['cleaned_sentence'], fn_q3_examples_df[f'proba {class_name}'])) + list(zip(fn_bottom_examples_df['original_sentence'], fn_bottom_examples_df['cleaned_sentence'], fn_bottom_examples_df[f'proba {class_name}']))

            if len(tp_examples_tuples) < 8 or len(fp_examples_tuples) < 4 or len(fn_examples_tuples) < 4:
                valid_sampling = False
                break

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

        if valid_sampling:
            print('Valid sampling. Done!')
            break
        
        # reset probas_df
        probas_df = probas_df_copy.copy()
        print('Invalid sampling. Trying again...')
        counter += 1

    if counter == 100:
        print('Could not sample valid examples. Please run train.py again.')
        return
    
    # save study samples to csv
    samples_df = pd.DataFrame(samples_dict)
    samples_df.to_csv('results/samples.csv')
    print('Samples saved to results/samples.csv')


if __name__ == '__main__':
    try:
        probas_df = pd.read_csv('results/probas.csv')
        scores_df = pd.read_csv('results/scores.csv')
        print('Sampling sentences...')
        _generate_samples_csvs(probas_df, scores_df)
    except FileNotFoundError as e:
        print('results/probas.csv and/or results/scores.csv not found. Please run train.py first')
