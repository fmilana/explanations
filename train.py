import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from augment import oversample
from classifier import MultiLabelProbClassifier


# generates confusion matrices for each class and saves to csv
def _generate_cm_csv(test_df, class_names, Y_pred, Y_true):
    for i, class_name in enumerate(class_names):
        # return boolean arrays for each category
        TP = np.logical_and(Y_pred[:, i] == 1, Y_true[:, i] == 1)
        FP = np.logical_and(Y_pred[:, i] == 1, Y_true[:, i] == 0)
        TN = np.logical_and(Y_pred[:, i] == 0, Y_true[:, i] == 0)
        FN = np.logical_and(Y_pred[:, i] == 0, Y_true[:, i] == 1)
        # get sentences for each category using boolean arrays as indices
        tp_sentences = test_df['original_sentence'][TP].tolist()
        fp_sentences = test_df['original_sentence'][FP].tolist()
        tn_sentences = test_df['original_sentence'][TN].tolist()
        fn_sentences = test_df['original_sentence'][FN].tolist()
        # get number of sentences
        num_tp_sentences = len(tp_sentences)
        num_fp_sentences = len(fp_sentences)
        num_tn_sentences = len(tn_sentences)
        num_fn_sentences = len(fn_sentences)
        # get maximum number of sentences in any category
        max_len = max(num_tp_sentences, num_fp_sentences, num_tn_sentences, num_fn_sentences)
        # pad shorter lists with empty strings to make all lists of equal length
        tp_sentences.extend([''] * (max_len - num_tp_sentences))
        fp_sentences.extend([''] * (max_len - num_fp_sentences))
        tn_sentences.extend([''] * (max_len - num_tn_sentences))
        fn_sentences.extend([''] * (max_len - num_fn_sentences))
        # create new df
        class_df = pd.DataFrame({
            f'true positives ({num_tp_sentences})': tp_sentences,
            f'false positives ({num_fp_sentences})': fp_sentences,
            f'true negatives ({num_tn_sentences})': tn_sentences,
            f'false negatives ({num_fn_sentences})': fn_sentences
        })
        # save to csv
        class_df.to_csv(f'results/cm/{class_name}_cm.csv', index=False)


def _generate_probas_csv(test_df, class_names, Y_test_prob):
    proba_df = pd.DataFrame(Y_test_prob, columns=[f'proba {class_names}' for i, class_names in enumerate(class_names)])
    test_df = test_df.reset_index(drop=True)
    test_df = pd.concat([test_df, proba_df], axis=1)
    test_df.to_csv('results/probas.csv', index=False)

    return test_df


def _generate_scores_csv(class_names, test_df, average_best_thresholds_per_class):
    index = ['threshold',
             'Q1 positive', 
             'median positive', 
             'Q3 positive', 
             'upper median',
             'lower median',
             'Q1 negative', 
             'median negative', 
             'Q3 negative']

    scores_df = pd.DataFrame(index=index, columns=class_names)

    scores_df.loc['threshold'] = [threshold for threshold in average_best_thresholds_per_class]

    scores_dict = {}
 
    for class_name, threshold in zip(class_names, average_best_thresholds_per_class):
        positive_class_df = test_df[test_df[f'pred {class_name}'] == 1]
        negative_class_df = test_df[test_df[f'pred {class_name}'] == 0]

        Q1_positive = positive_class_df[f'proba {class_name}'].quantile(0.25)
        median_positive = positive_class_df[f'proba {class_name}'].median()
        Q3_positive = positive_class_df[f'proba {class_name}'].quantile(0.75)
        Q1_negative = negative_class_df[f'proba {class_name}'].quantile(0.25)
        median_negative = negative_class_df[f'proba {class_name}'].median()
        Q3_negative = negative_class_df[f'proba {class_name}'].quantile(0.75)
        mean_negative = negative_class_df[f'proba {class_name}'].mean()

        upper_median_df = positive_class_df[(positive_class_df[f'proba {class_name}'] >= threshold) & 
                                            (positive_class_df[f'proba {class_name}'] <= Q1_positive)]
        upper_median = upper_median_df[f'proba {class_name}'].median()

        lower_median_df = negative_class_df[(negative_class_df[f'proba {class_name}'] >= Q3_negative) & 
                                            (negative_class_df[f'proba {class_name}'] <= threshold)]
        lower_median = lower_median_df[f'proba {class_name}'].median()

        scores_dict[class_name] = {
            'Q1 positive': Q1_positive,
            'median positive': median_positive,
            'Q3 positive': Q3_positive,
            'upper median': upper_median,
            'lower median': lower_median,
            'Q1 negative': Q1_negative,
            'median negative': median_negative,
            'Q3 negative': Q3_negative
        }

    for class_name in class_names:
        scores_df.loc['Q1 positive', class_name] = scores_dict[class_name]['Q1 positive']
        scores_df.loc['median positive', class_name] = scores_dict[class_name]['median positive']
        scores_df.loc['Q3 positive', class_name] = scores_dict[class_name]['Q3 positive']
        scores_df.loc['upper median', class_name] = scores_dict[class_name]['upper median']
        scores_df.loc['lower median', class_name] = scores_dict[class_name]['lower median']
        scores_df.loc['Q1 negative', class_name] = scores_dict[class_name]['Q1 negative']
        scores_df.loc['median negative', class_name] = scores_dict[class_name]['median negative']
        scores_df.loc['Q3 negative', class_name] = scores_dict[class_name]['Q3 negative']
                                                                        
    scores_df.to_csv('results/scores.csv')


def _train_and_validate(df):
    # remove rows with null values in cleaned_sentence
    df = df[df['cleaned_sentence'].notnull()]

    # get list of class names
    class_names = df.columns[7:].tolist()
    # process sentence embedding strings
    df.loc[:, 'sentence_embedding'] = df['sentence_embedding'].apply(
        lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(',', ' ')
            .replace(']',''), sep=' '
        )
    )
    # create list of review ids for training
    review_ids = df['review_id'].unique().tolist()
    # set test size
    test_size = 5 # arbitrary
    # randomly select 5 test ids
    random.seed(42)
    test_ids = random.sample(review_ids, test_size)
    # create test df based on test ids
    test_df = df[df['review_id'].isin(test_ids)]
    # remove test ids from review ids
    review_ids = [id for id in review_ids if id not in test_ids]
    # filter df to exclude test ids
    df = df[~df['review_id'].isin(test_ids)]
    # set thresholds for cross-validation
    thresholds = np.arange(0.1, 1, 0.1)

    # initialise 2D NumPy arrays (iterations x classes) to store best thresholds and scores for each class 
    best_thresholds_per_class = np.zeros((len(review_ids), len(class_names)))
    best_scores_per_class = np.zeros((len(review_ids), len(class_names)))

    for iteration_index, review_id in enumerate(review_ids):
        print(f'==> Training on split {iteration_index+1}/{len(review_ids)}...')
        # create validation set
        validation_df = df[df['review_id'] == review_id]
        # create training set by removing test and validation sets
        train_df = df[~df['review_id'].isin(test_ids + [review_id])]
        # prepare training data
        X_train = np.array(train_df['sentence_embedding'].tolist())
        Y_train = np.array(train_df.iloc[:, 7:])
        # # oversample minority classes (if present)
        X_train, Y_train = oversample(X_train, Y_train)
        # prepare validation data
        X_validation = np.array(validation_df['sentence_embedding'].tolist())
        Y_validation = np.array(validation_df.iloc[:, 7:])

        # reinitialise classifier for each split
        clf = MultiLabelProbClassifier()
        # fit classifier
        clf.fit(X_train, Y_train)
        # predict probabilities on validation set
        Y_prob = clf.predict_proba(X_validation)

        # iterate over each class (index)
        for class_index in range(len(class_names)):
            # get true labels for the current class (1D array)
            y_class_true = Y_validation[:, class_index]
            # initialise best score and threshold for the current class
            best_score_for_class = 0
            best_threshold_for_class = 0
            # iterate over thresholds
            for threshold in thresholds:
                # convert probabilities to binary predictions for the current class
                y_class_pred = (Y_prob[:, class_index] > threshold).astype(int)
                # calculate f1 score for the current class (compare 1D arrays)
                score = f1_score(y_class_true, y_class_pred)
                # update best score and threshold if score is better
                if score > best_score_for_class:
                    best_score_for_class = score
                    best_threshold_for_class = threshold

            # store the best threshold and score for the current class
            best_thresholds_per_class[iteration_index, class_index] = best_threshold_for_class
            best_scores_per_class[iteration_index, class_index] = best_score_for_class

        print('done.')

    # calculate the average best threshold and scores for each class (average of each column)
    average_best_thresholds_per_class = np.mean(best_thresholds_per_class, axis=0)
    average_scores_per_class = np.mean(best_scores_per_class, axis=0)

    print(f'===============VALIDATION RESULTS===============')
    print(f'average best thresholds per class: {average_best_thresholds_per_class}')
    print(f'average scores per class: {average_scores_per_class}')
    print(f'average score across all classes: {np.mean(average_scores_per_class)}')
    print(f'================================================')

    # train the classifier on the full training set
    X_train_full = np.array(df['sentence_embedding'].tolist())
    Y_train_full = np.array(df.iloc[:, 7:])
    clf.fit(X_train_full, Y_train_full)
    # prepare test data
    X_test = np.array(test_df['sentence_embedding'].tolist())
    Y_test = np.array(test_df.iloc[:, 7:])
    # predict probabilities on the test set
    Y_test_prob = clf.predict_proba(X_test)
    # create a 2D NumPy array of zeros with same size as Y_test_prob
    Y_base_test_pred = np.zeros_like(Y_test_prob, dtype=int)
    Y_test_pred = np.zeros_like(Y_test_prob, dtype=int)
    # apply the average best threshold from cross-validation for each class
    for class_index in range(len(class_names)):
        Y_base_test_pred[:, class_index] = (Y_test_prob[:, class_index] >= 0.5).astype(int)
        Y_test_pred[:, class_index] = (Y_test_prob[:, class_index] >= average_best_thresholds_per_class[class_index]).astype(int)
    # append predictions to test_df and write to csv
    for i, class_name in enumerate(class_names):
        test_df[f'pred {class_name}'] = Y_test_pred[:, i]
    test_df.to_csv('results/output.csv', index=False)

    # create 1D NumPy arrays to store test scores for each class
    test_scores_per_class = np.zeros((len(class_names)))
    base_test_scores_per_class = np.zeros((len(class_names)))
    # calculate test scores for each class
    for i, class_name in enumerate(class_names):
        # using 0.5 threshold
        base_test_scores_per_class[i] = f1_score(Y_test[:, i], Y_base_test_pred[:, i])
        # using average best thresholds
        test_scores_per_class[i] = f1_score(Y_test[:, i], Y_test_pred[:, i])
    # calculate overall test score using 0.5 threshold
    base_test_score = f1_score(Y_test, Y_base_test_pred, average='weighted')
    # calculate overall test score using average best thresholds
    test_score = f1_score(Y_test, Y_test_pred, average='weighted')

    with open('results/model_scores.txt', 'w') as f:
        output = f'===============TEST RESULTS===============\n'
        output += f'test F1 Scores per class using 0.5 threshold: {base_test_scores_per_class}\n'
        output += f'test F1 Scores per class using average best thresholds: {test_scores_per_class}\n'
        output += f'base test F1 Score using 0.5 threshold: {base_test_score}\n'
        output += f'test F1 Score using average best thresholds ({average_best_thresholds_per_class}): {test_score}\n'
        output += f'==========================================\n'
        print(output)
        f.write(output)

    # write confusion matrices to csv's
    _generate_cm_csv(test_df, class_names, Y_test_pred, Y_test)
    # write test_df probas to csv
    test_df = _generate_probas_csv(test_df, class_names, Y_test_prob)
    # write scores to csv
    _generate_scores_csv(class_names, test_df, average_best_thresholds_per_class)

    return clf


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/train.csv')
        print('Training and validating model...')
        clf = _train_and_validate(df)
        # save the model to disk
        joblib.dump(clf, 'model/model.sav')
        print('Done. Model saved in model/model.sav.')
    except FileNotFoundError as e:
        print('data/train.csv not found.')
