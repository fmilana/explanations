import json
import random
import GPUtil
import xgboost
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from augment import oversample


def _get_device():
    if GPUtil.getGPUs():
        print('Using GPU for training.')
        return 'cuda'
    else:
        print('Using CPU for training.')
        return 'cpu'


def _train_and_validate(df):
    # Remove rows with null values in cleaned_sentence
    df = df[df['cleaned_sentence'].notnull()]

    # Get list of class names
    class_names = df.columns[7:].tolist()

    # Process sentence embedding strings
    df.loc[:, 'sentence_embedding'] = df['sentence_embedding'].apply(
        lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(',', ' ')
            .replace(']',''), sep=' '
        )
    )

    # Prepare data
    X = np.array(df['sentence_embedding'].tolist())
    Y = np.array(df.iloc[:, 7:])
    
    groups = df['review_id'].values  # Use review_id for grouping in cross-validation

    num_classes = Y.shape[1]

    # Split data into training and test sets
    unique_review_ids = df['review_id'].unique()
    random.seed(42)
    test_review_ids = random.sample(list(unique_review_ids), 5) # 5 random review_ids for test set
    train_review_ids = [review_id for review_id in unique_review_ids if review_id not in test_review_ids]

    train_mask = df['review_id'].isin(train_review_ids)
    test_mask = df['review_id'].isin(test_review_ids)

    X_train, X_test = X[train_mask], X[test_mask]
    Y_train, Y_test = Y[train_mask], Y[test_mask]
    groups_train = groups[train_mask]

    device = _get_device()

    # Define parameter grid
    param_grid = {
        # 'max_depth': [4, 5, 6],
        'max_depth': [2, 3],
        # 'learning_rate': [0.01, 0.1, 0.3],
        'learning_rate': [0.1, 0.3, 0.5],
        # 'subsample': [0.8, 1.0],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [1, 10, 100]
    }

    # Set up GroupKFold for cross-validation
    n_splits = 4
    group_kfold = GroupKFold(n_splits=n_splits)

    best_score = 0
    best_params = None

    # cross-validation loop
    for fold_counter, (train_index, val_index) in enumerate(group_kfold.split(X_train, Y_train, groups_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

        # Oversample minority class (if needed)
        X_train_fold, Y_train_fold = oversample(X_train_fold, Y_train_fold)

        # Create QuantileDMatrix
        dtrain = xgboost.QuantileDMatrix(X_train_fold, label=Y_train_fold)
        dval = xgboost.QuantileDMatrix(X_val_fold, label=Y_val_fold)

        for params in [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]:
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'device': device
            })

            params_key = str(params)

            # Train the model
            num_round = 100  # adjust this
            clf = xgboost.train(params, dtrain, num_round, evals=[(dval, 'eval')], early_stopping_rounds=10)

            # Predict on validation set
            y_pred_proba = clf.predict(dval)

            y_pred = (y_pred_proba >= 0.5).astype(int)
            score = f1_score(Y_val_fold, y_pred, average='weighted')

            if score > best_score:
                best_score = score
                best_params = params

    print('Best hyperparameters:', best_params)
    print('Best cross-validation score:', best_score)

    # Train final model on all training data
    # Oversample minority class (if needed)
    X_train, Y_train = oversample(X_train, Y_train)
    # Create QuantileDMatrix
    dtrain_full = xgboost.QuantileDMatrix(X_train, label=Y_train)
    final_model = xgboost.train(best_params, dtrain_full, num_round)

    # Evaluate final model on test set
    dtest = xgboost.QuantileDMatrix(X_test, label=Y_test)
    Y_prob_test = final_model.predict(dtest)
    Y_pred_test = (Y_prob_test >= 0.5).astype(int)

    # Calculate and print final F1 score on test set
    final_score_test = f1_score(Y_test, Y_pred_test, average='weighted')
    print('Final F1 Score on Test Set:', final_score_test)

    # get train and test dfs and save to csv
    df[train_mask].to_csv('data/train.csv', index=False)
    df[test_mask].to_csv('data/test.csv', index=False)

    # save model to disk
    final_model.save_model('model/xgb_model.json')


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/data.csv')
        print('Training and validating model...')
        _train_and_validate(df)
        print('Done.') 
        print('Train data saved in data/train.csv')
        print('Test data saved in data/test.csv.')
        print('Model saved in model/')
    except FileNotFoundError as e:
        print('data/data.csv not found.')
