import random
import pandas as pd


def _generate_samples_csvs(probas_df, scores_df):
    # remove sentences from probas_df with 1 or less words when cleaned
    probas_df = probas_df[probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) > 1)]
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
            # get scores
            q1_score_positive = scores_df.loc['Q1 positive', class_name]
            upper_median_positive = scores_df.loc['upper median', class_name]
            lower_median_negative = scores_df.loc['lower median', class_name]
            q3_score_negative = scores_df.loc['Q3 negative', class_name]

            # queries

            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]

            # sample 3 queries from Q1 false positives (closest to q1 positive)
            q1_fp_query_df = false_positive_df.loc[(false_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(3).index]
            q1_fp_query_tuples = []
            for _, row in q1_fp_query_df.iterrows():
                q1_fp_query_tuples.append((row['original_sentence'], row['cleaned_sentence'], row[f'proba {class_name}']))
            # remove sampled query from probas_df
            probas_df = probas_df.drop(q1_fp_query_df.index)

            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]

            # sample 3 queries from upper median false positives (closest to median between median and q1 positive)
            upper_median_fp_query_df = false_positive_df.loc[(false_positive_df[f'proba {class_name}'] - upper_median_positive).abs().nsmallest(3).index]
            upper_median_fp_query_tuples = []
            for _, row in upper_median_fp_query_df.iterrows():
                upper_median_fp_query_tuples.append((row['original_sentence'], row['cleaned_sentence'], row[f'proba {class_name}']))
            # remove sampled query from probas_df
            probas_df = probas_df.drop(upper_median_fp_query_df.index)

            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]

            # sample 3 queries from lower median false negatives (closest to median between median and q3 negative)
            lower_median_fn_query_df = false_negative_df.loc[(false_negative_df[f'proba {class_name}'] - lower_median_negative).abs().nsmallest(3).index]
            lower_median_fn_query_tuples = []
            for _, row in lower_median_fn_query_df.iterrows():
                lower_median_fn_query_tuples.append((row['original_sentence'], row['cleaned_sentence'], row[f'proba {class_name}']))
            # remove sampled query from probas_df
            probas_df = probas_df.drop(lower_median_fn_query_df.index)

            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]

            # sample 3 queries from Q3 false negatives (closest to q3 negative)
            q3_fn_query_df = false_negative_df.loc[(false_negative_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(3).index]
            q3_fn_query_tuples = []
            for _, row in q3_fn_query_df.iterrows():
                q3_fn_query_tuples.append((row['original_sentence'], row['cleaned_sentence'], row[f'proba {class_name}']))
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(q3_fn_query_df.index)

            # examples

            # get true positive dfs
            true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 1)]
            # try to sample 12 examples from top true positives (at random among top 24 true positives)
            num_samples = min(12, len(true_positive_df.nlargest(24, f'proba {class_name}')))
            tp_top_examples_df = true_positive_df.nlargest(24, f'proba {class_name}').sample(n=num_samples)
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(tp_top_examples_df.index)

            # get true positive dfs again
            true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 1)]
            # sample 12 examples from around q1 positives (closest to q1 positive)
            tp_q1_examples_df = true_positive_df.loc[(true_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(12).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(tp_q1_examples_df.index)

            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
            
            # get false positive df
            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
            # try to sample 6 examples from top false positives (at random among top 18 false positives)
            num_samples = min(6, len(false_positive_df.nlargest(12, f'proba {class_name}')))
            fp_top_examples_df = false_positive_df.nlargest(12, f'proba {class_name}').sample(n=num_samples)
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fp_top_examples_df.index)

            # get false positive df again
            false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f'pred {class_name}'] == 1)]
            # sample 6 examples from around q1 false positives (closest to q1 positive)
            fp_q1_examples_df = false_positive_df.loc[(false_positive_df[f'proba {class_name}'] - q1_score_positive).abs().nsmallest(6).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fp_q1_examples_df.index)

            # get false negative df
            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
            # try to sample 6 examples from bottom false negatives
            num_samples = min(6, len(false_negative_df.nsmallest(12, f'proba {class_name}')))
            fn_bottom_examples_df = false_negative_df.nsmallest(12, f'proba {class_name}').sample(n=num_samples)
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fn_bottom_examples_df.index)

            # get false negative df again
            false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f'pred {class_name}'] == 0)]
            # try to sample 6 examples from around q3 false negatives (closest to q3 negative)
            fn_q3_examples_df = false_negative_df.loc[(false_negative_df[f'proba {class_name}'] - q3_score_negative).abs().nsmallest(6).index]
            # remove sampled queries from probas_df
            probas_df = probas_df.drop(fn_q3_examples_df.index)
            
            # create list of tuples (sentences, proba)
            tp_examples_dict = {
                'Top': list(zip(tp_top_examples_df['original_sentence'], tp_top_examples_df['cleaned_sentence'], tp_top_examples_df[f'proba {class_name}'])),
                'Q1': list(zip(tp_q1_examples_df['original_sentence'], tp_q1_examples_df['cleaned_sentence'], tp_q1_examples_df[f'proba {class_name}']))
            }
            fp_examples_dict = {
                'Top': list(zip(fp_top_examples_df['original_sentence'], fp_top_examples_df['cleaned_sentence'], fp_top_examples_df[f'proba {class_name}'])),
                'Q1': list(zip(fp_q1_examples_df['original_sentence'], fp_q1_examples_df['cleaned_sentence'], fp_q1_examples_df[f'proba {class_name}']))
            }
            fn_examples_dict = {
                'Q3': list(zip(fn_q3_examples_df['original_sentence'], fn_q3_examples_df['cleaned_sentence'], fn_q3_examples_df[f'proba {class_name}'])),
                'Bottom': list(zip(fn_bottom_examples_df['original_sentence'], fn_bottom_examples_df['cleaned_sentence'], fn_bottom_examples_df[f'proba {class_name}']))
            }

            if len(tp_examples_dict['Top'] + tp_examples_dict['Q1']) < 8 or len(fp_examples_dict['Top'] + fp_examples_dict['Q1']) < 4 or len(fn_examples_dict['Q3'] + fn_examples_dict['Bottom']) < 4:
                valid_sampling = False
                print(f"Invalid sampling for class {class_name}:\n{len(tp_examples_dict['Top'] + tp_examples_dict['Q1'])} TP examples\n{len(fp_examples_dict['Top'] + fp_examples_dict['Q1'])} FP examples\n{len(fn_examples_dict['Q3'] + fn_examples_dict['Bottom'])} FN examples")
                break

            samples_dict[class_name] = {
                'TP Examples Dict': tp_examples_dict,
                'FP Examples Dict': fp_examples_dict,
                'FN Examples Dict': fn_examples_dict,
                'Q1 False Positive Query Tuples': q1_fp_query_tuples,
                'Upper Median False Positive Query Tuples': upper_median_fp_query_tuples,
                'Lower Median False Negative Query Tuples': lower_median_fn_query_tuples,
                'Q3 False Negative Query Tuples': q3_fn_query_tuples
            }

            # reset probas_df so that there are enough examples for the next class
            probas_df = probas_df_copy.copy()

        if valid_sampling:
            print('Valid sampling. Done!')
            break

        # invalid sampling
        counter += 1

    if counter == 100:
        print('Could not sample enough examples. Please run train.py again.')
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
