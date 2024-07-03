import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from augment import oversample
from classifier import MultiLabelProbClassifier
from sklearn.model_selection import KFold


# hardcoded thresholds for each class
# BEST_TRAIN_THRESHOLDS = [0.87, 0.68, 0.79, 0.78]
# BEST_TEST_THRESHOLDS = [0.3, 0.31, 0.25, 0.23]


# generates confusion matrices for each class and saves to csv
def _generate_cm_csv(df, df_name, split_index, class_names, Y_pred, Y_true):
    for i, class_name in enumerate(class_names):
        # return boolean arrays for each category
        TP = np.logical_and(Y_pred[:, i] == 1, Y_true[:, i] == 1)
        FP = np.logical_and(Y_pred[:, i] == 1, Y_true[:, i] == 0)
        TN = np.logical_and(Y_pred[:, i] == 0, Y_true[:, i] == 0)
        FN = np.logical_and(Y_pred[:, i] == 0, Y_true[:, i] == 1)
        # get sentences for each category using boolean arrays as indices
        tp_sentences = df['original_sentence'][TP].tolist()
        fp_sentences = df['original_sentence'][FP].tolist()
        tn_sentences = df['original_sentence'][TN].tolist()
        fn_sentences = df['original_sentence'][FN].tolist()
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
        if not os.path.exists(f'results/{df_name}/{split_index + 1}/cm/'):
            os.makedirs(f'results/{df_name}/{split_index + 1}/cm/')
        class_df.to_csv(f'results/{df_name}/{split_index + 1}/cm/{class_name}_cm.csv', index=False)


def _add_pred_and_proba_to_csv(df, df_name, split_index, class_names, Y_prob, best_thresholds):
    # create a 2D NumPy array of zeros with same size as Y_prob
    Y_base_pred = np.zeros_like(Y_prob, dtype=int)
    Y_pred = np.zeros_like(Y_prob, dtype=int)
    # apply the best threshold from cross-validation for each class
    for class_index in range(len(class_names)):
        Y_base_pred[:, class_index] = (Y_prob[:, class_index] >= 0.5).astype(int)
        Y_pred[:, class_index] = (Y_prob[:, class_index] >= best_thresholds[class_index]).astype(int)

    proba_df = pd.DataFrame(Y_prob, columns=[f'proba {class_names}' for i, class_names in enumerate(class_names)])
    df = df.reset_index(drop=True)
    df = pd.concat([df, proba_df], axis=1)
    # save to csv
    if not os.path.exists(f'results/{df_name}/{split_index + 1}/'):
        os.makedirs(f'results/{df_name}/{split_index + 1}/')
    df.to_csv(f'results/{df_name}/{split_index + 1}/probas.csv', index=False)

    return Y_base_pred, Y_pred


def _generate_metrics_txt(df_name, split_index, Y, Y_base_pred, Y_pred, class_names, best_thresholds):
    # create 1D NumPy arrays to store test scores for each class
    scores_per_class = np.zeros((len(class_names)))
    base_scores_per_class = np.zeros((len(class_names)))
    # calculate test scores for each class
    for i, class_name in enumerate(class_names):
        # using 0.5 threshold
        base_scores_per_class[i] = f1_score(Y[:, i], Y_base_pred[:, i])
        # using best thresholds
        scores_per_class[i] = f1_score(Y[:, i], Y_pred[:, i])
    # calculate overall test score using 0.5 threshold
    base_score = f1_score(Y, Y_base_pred, average='weighted')
    # calculate overall test score using best thresholds
    score = f1_score(Y, Y_pred, average='weighted')

    # save to txt
    if not os.path.exists(f'results/{df_name}/{split_index + 1}/'):
        os.makedirs(f'results/{df_name}/{split_index + 1}/')
    with open(f'results/{df_name}/{split_index + 1}/model_scores.txt', 'w') as f:
        output = f'==============={df_name} {split_index + 1} RESULTS===============\n'
        output += f'{df_name} F1 Scores per class using 0.5 threshold: {base_scores_per_class}\n'
        output += f'{df_name} F1 Scores per class using best thresholds: {scores_per_class}\n'
        output += f'base {df_name} F1 Score using 0.5 threshold: {base_score}\n'
        output += f'{df_name} F1 Score using best thresholds ({best_thresholds}): {score}\n'
        output += f'==========================================\n'
        print(output)
        f.write(output)


def _generate_class_scores_csv(df, df_name, split_index, class_names, best_thresholds):
    index = ['threshold',
             'upper midpoint',
             'lower midpoint']

    scores_df = pd.DataFrame(index=index, columns=class_names)

    scores_df.loc['threshold'] = [threshold for threshold in best_thresholds]
 
    for class_name, threshold in zip(class_names, best_thresholds):
        scores_df.loc['upper midpoint', class_name] = (threshold + 1) / 2
        scores_df.loc['lower midpoint', class_name] = threshold / 2

    # save to csv
    if not os.path.exists(f'results/{df_name}/{split_index + 1}/'):
        os.makedirs(f'results/{df_name}/{split_index + 1}/')                                   
    scores_df.to_csv(f'results/{df_name}/{split_index + 1}/scores.csv')


def _train_all_splits(df):
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
    n_splits = 7 # 21 / 3 = 7
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for split_index, (train_index, val_index) in enumerate(kf.split(review_ids)):
        print(f'==> Training on split {split_index + 1}/{n_splits}...')

        train_ids = [review_ids[i] for i in train_index]
        val_ids = [review_ids[i] for i in val_index]

        train_df = df[df['review_id'].isin(train_ids)]
        test_df = df[df['review_id'].isin(val_ids)] # we call validation set test for consistency with previous code

        # prepare training data
        X_train = np.array(train_df['sentence_embedding'].tolist())
        Y_train = np.array(train_df.iloc[:, 7:])
        # # oversample minority classes (if present)
        X_train_oversampled, Y_train_oversampled = oversample(X_train, Y_train)
        # prepare validation ("test") data
        X_test = np.array(test_df['sentence_embedding'].tolist())
        Y_test = np.array(test_df.iloc[:, 7:])

        # reinitialise classifier for each split
        clf = MultiLabelProbClassifier()
        # fit classifier
        print('Fitting classifier...')
        clf.fit(X_train_oversampled, Y_train_oversampled)
        print('Done.')
        # predict probabilities on validation set
        print('Predicting probabilities on test set...')
        Y_prob = clf.predict_proba(X_test)
        print('Done.')

        # find best threshold for each class
        # set thresholds to try for classification
        thresholds = np.arange(0.1, 1, 0.1)
        # initialise 2D NumPy arrays (iterations x classes) to store best thresholds and scores for each class 
        best_thresholds = np.zeros(len(class_names))
        best_scores = np.zeros(len(class_names))

        print('Finding best thresholds according to split...')
        # iterate over each class (index)
        for class_index in range(len(class_names)):
            # get true labels for the current class (1D array)
            y_class_true = Y_test[:, class_index]
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
            best_thresholds[class_index] = best_threshold_for_class
            best_scores[class_index] = best_score_for_class

        print('Done.')
        print(f'===============SPLIT {split_index + 1} RESULTS===============')
        print(f'best thresholds per class: {best_thresholds}')
        print(f'scores per class: {best_scores}')
        print(f'score across all classes: {np.mean(best_scores)}')
        print(f'================================================')

        print('Predicting probabilities on test set...')
        Y_test_prob = clf.predict_proba(X_test)

        # add predictions and probabilities to test df and save to csv
        Y_base_test_pred, Y_test_pred = _add_pred_and_proba_to_csv(test_df, 'test', split_index, class_names, Y_test_prob, best_thresholds)

        # calculate and write test scores using 0.5 threshold and best thresholds
        _generate_metrics_txt('test', split_index, Y_base_test_pred, Y_test_pred, Y_test, class_names, best_thresholds)

        # write test confusion matrices to csv's
        _generate_cm_csv(test_df, 'test', split_index, class_names, Y_test_pred, Y_test)
        
        # write class scores to csv
        _generate_class_scores_csv(test_df, 'test', split_index, class_names, best_thresholds)

        # predict on training set from which to sample example sentences later
        Y_train_prob = clf.predict_proba(X_train)

        Y_base_train_pred, Y_train_pred = _add_pred_and_proba_to_csv(train_df, 'train', split_index, class_names, Y_train_prob, best_thresholds)

        _generate_metrics_txt('train', split_index, Y_base_train_pred, Y_train_pred, Y_train, class_names, best_thresholds)

        _generate_cm_csv(train_df, 'train', split_index, class_names, Y_train_pred, Y_train)

        _generate_class_scores_csv(train_df, 'train', split_index, class_names, best_thresholds)

        # save model to disk
        joblib.dump(clf, f'model/model_{split_index + 1}.sav')


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/train.csv')
        print('Training model on all splits...')
        _train_all_splits(df)
        print('Done. Models saved in model/')
    except FileNotFoundError as e:
        print('data/train.csv not found.')
