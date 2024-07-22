import json
import xgboost
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def _add_pred_and_prob_to_csv(df, df_name, class_names, Y_prob):
    # create a 2D NumPy array of zeros with same size as Y_prob
    Y_pred = np.zeros_like(Y_prob, dtype=int)

    for class_index, class_name in enumerate(class_names):
        Y_pred[:, class_index] = (Y_prob[:, class_index] >= 0.5).astype(int)

    pred_df = pd.DataFrame(Y_pred, columns=[f'pred {class_names}' for i, class_names in enumerate(class_names)])
    df = df.reset_index(drop=True)
    df = pd.concat([df, pred_df], axis=1)

    prob_df = pd.DataFrame(Y_prob, columns=[f'proba {class_names}' for i, class_names in enumerate(class_names)])
    df = pd.concat([df, prob_df], axis=1)
    
    df.to_csv(f'results/{df_name}/probas.csv', index=False)

    return Y_pred


def _generate_metrics_txt(df_name, Y, Y_pred, class_names):
    # create 1D NumPy arrays to store test scores for each class
    scores_per_class = np.zeros((len(class_names)))
    # calculate test scores for each class
    for i, class_name in enumerate(class_names):
        scores_per_class[i] = f1_score(Y[:, i], Y_pred[:, i])
    # calculate overall test score
    score = f1_score(Y, Y_pred, average='weighted')
    # calculate overall test score
    score = f1_score(Y, Y_pred, average='weighted')

    with open(f'results/{df_name}/model_scores.txt', 'w') as f:
        output = f'==============={df_name} RESULTS===============\n'
        output += f'{df_name} F1 Scores per class: {scores_per_class}\n'
        output += f'base {df_name} F1 Score: {score}\n'
        output += f'==========================================\n'
        print(output)
        f.write(output)


# generates confusion matrices for each class and saves to csv
def _generate_cm_csv(df, df_name, class_names, Y_pred, Y_true):
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
        class_df.to_csv(f'results/{df_name}/cm/{class_name}_cm.csv', index=False)


def _predict_set(df, df_name, clf):
    X = np.array(df['sentence_embedding'].tolist())
    Y = np.array(df.iloc[:, 5:])

    class_names = df.columns[5:].tolist()

    # predict probabilities on the test set
    dx = xgboost.DMatrix(X) # create DMatrix
    Y_prob = clf.predict(dx)
    # add predictions and probabilities to test df and save to csv
    Y_pred = _add_pred_and_prob_to_csv(df, df_name, class_names, Y_prob)
    # calculate and write test scores using 0.5 threshold
    _generate_metrics_txt(df_name, Y_pred, Y, class_names)
    # write test confusion matrices to csv's
    _generate_cm_csv(df, df_name, class_names, Y_pred, Y)


if __name__ == '__main__':
    try:
        train_df = pd.read_csv('data/train.csv')
        train_df.loc[:, 'sentence_embedding'] = train_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )
        test_df = pd.read_csv('data/test.csv')
        test_df.loc[:, 'sentence_embedding'] = test_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(',', ' ')
                .replace(']',''), sep=' '
            )
        )
        clf = xgboost.Booster()
        clf.load_model('model/xgb_model.json')

        print('Predicting on test set...')
        _predict_set(test_df, 'test', clf)
        print('Done. Results saved in results/test/')
        print('Predicting on train set...')
        _predict_set(train_df, 'train', clf)
        print('Done. Results saved in results/train/')
    except FileNotFoundError as e:
        print('One or more of these files are missing:')
        print('data/train.csv')
        print('data/test.csv')
        print('model/xgboost.json')
        print('Please run train.py first.')
        exit()