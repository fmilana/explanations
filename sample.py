import random
import pandas as pd


def sample_sentences():
    probas_df = pd.read_csv("results/probas.csv")
    # remove sentences with less than 5 words (when cleaned)
    probas_df = probas_df[probas_df['cleaned_sentence'].apply(lambda x: len(x.split()) >= 5)]
    probas_df = probas_df.reset_index(drop=True)
    scores_df = pd.read_csv("results/scores.csv")
    scores_df.set_index(scores_df.columns[0], inplace=True)

    class_names = [class_name for class_name in probas_df.columns[7:].tolist() if not (class_name.startswith("pred") or class_name.startswith("proba"))]

    sample_dict = {}

    for class_name in class_names:
        q1_positive = scores_df.loc["Q1 positive", class_name]
        q3_negative = scores_df.loc["Q3 negative", class_name]
        # sample query from top 10 positive
        top_positive_query_df = probas_df.nlargest(10, f"proba {class_name}").sample(n=1)
        top_positive_query_tuple = (top_positive_query_df["original_sentence"].values[0], top_positive_query_df["cleaned_sentence"].values[0], top_positive_query_df[f"proba {class_name}"].values[0])
        # sample query from around q1 positive
        q1_positive_query_df = probas_df[abs(probas_df[f"proba {class_name}"] - q1_positive) <= 0.05].sample(n=1)
        q1_positive_query_tuple = (q1_positive_query_df["original_sentence"].values[0], q1_positive_query_df["cleaned_sentence"].values[0], q1_positive_query_df[f"proba {class_name}"].values[0])
        # sample query from around q3 negative
        q3_negative_query_df = probas_df[abs(probas_df[f"proba {class_name}"] - q3_negative) <= 0.05].sample(n=1)
        q3_negative_query_tuple = (q3_negative_query_df["original_sentence"].values[0], q3_negative_query_df["cleaned_sentence"].values[0], q3_negative_query_df[f"proba {class_name}"].values[0])
        # sample query from bottom 10 negative
        bottom_negative_query_df = probas_df.nsmallest(10, f"proba {class_name}").sample(n=1)
        bottom_negative_query_tuple = (bottom_negative_query_df["original_sentence"].values[0], bottom_negative_query_df["cleaned_sentence"].values[0], bottom_negative_query_df[f"proba {class_name}"].values[0])
        # remove sampled query from probas_df
        for df in [top_positive_query_df, q1_positive_query_df, q3_negative_query_df, bottom_negative_query_df]:
            probas_df = probas_df.merge(df, how='outer', indicator=True)
            probas_df = probas_df[probas_df['_merge'] == 'left_only']
            del probas_df['_merge']
        # get true positive, false positive, and false negative dfs
        true_positive_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f"pred {class_name}"] == 1)]
        false_positive_df = probas_df[(probas_df[class_name] == 0) & (probas_df[f"pred {class_name}"] == 1)]
        false_negative_df = probas_df[(probas_df[class_name] == 1) & (probas_df[f"pred {class_name}"] == 0)]
        # sample 4 examples from top true positives
        tp_top_examples_df = true_positive_df.nlargest(4, f"proba {class_name}")
        # sample 4 examples from around q1 positives (closest to q1 positive)
        tp_q1_examples_df = true_positive_df.loc[(true_positive_df[f"proba {class_name}"] - q1_positive).abs().nsmallest(4).index]
        # sample 2 examples from top false positives
        fp_top_examples_df = false_positive_df.nlargest(2, f"proba {class_name}")
        # sample 2 examples from around q1 false positives (closest to q1 positive)
        fp_q1_examples_df = false_positive_df.loc[(false_positive_df[f"proba {class_name}"] - q1_positive).abs().nsmallest(2).index]
        # sample 2 examples from around q3 false negatives (closest to q3 negative)
        fn_q3_examples_df = false_negative_df.loc[(false_negative_df[f"proba {class_name}"] - q3_negative).abs().nsmallest(2).index]
        # sample 2 examples from bottom false negatives
        fn_bottom_examples_df = false_negative_df.nsmallest(2, f"proba {class_name}")
        # create list of tuples (sentences, proba)
        tp_examples_tuples = list(zip(tp_top_examples_df['original_sentence'], tp_top_examples_df['cleaned_sentence'], tp_top_examples_df[f'proba {class_name}'])) + list(zip(tp_q1_examples_df['original_sentence'], tp_q1_examples_df['cleaned_sentence'], tp_q1_examples_df[f'proba {class_name}']))
        fp_examples_tuples = list(zip(fp_top_examples_df['original_sentence'], fp_top_examples_df['cleaned_sentence'], fp_top_examples_df[f'proba {class_name}'])) + list(zip(fp_q1_examples_df['original_sentence'], fp_q1_examples_df['cleaned_sentence'], fp_q1_examples_df[f'proba {class_name}']))
        fn_examples_tuples = list(zip(fn_q3_examples_df['original_sentence'], fn_q3_examples_df['cleaned_sentence'], fn_q3_examples_df[f'proba {class_name}'])) + list(zip(fn_bottom_examples_df['original_sentence'], fn_bottom_examples_df['cleaned_sentence'], fn_bottom_examples_df[f'proba {class_name}']))
        # shuffle lists
        # random.shuffle(tp_examples_tuples)
        # random.shuffle(fp_examples_tuples)
        # random.shuffle(fn_examples_tuples)
        # add to sample_dict
        sample_dict[class_name] = [
            top_positive_query_tuple, 
            q1_positive_query_tuple, 
            q3_negative_query_tuple, 
            bottom_negative_query_tuple, 
            tp_examples_tuples, 
            fp_examples_tuples, 
            fn_examples_tuples
            ]

    return sample_dict
