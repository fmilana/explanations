import os
import shutil
import random
import torch
import numpy as np
import pandas as pd
import optuna
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction
)
from sklearn.model_selection import GroupKFold
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


MODEL_CKPT = 'distilbert-base-uncased'
# MODEL_CKPT = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')
LABEL_NAMES = []

best_thresholds = []


def _generate_cm_csv(test_df, Y_pred, Y_true):
    for i, label in enumerate(LABEL_NAMES):
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
        class_df.to_csv(f'results/cm/{label}_cm.csv', index=False)


def _generate_probas_csv(test_df, y_test_probas):
    proba_df = pd.DataFrame(y_test_probas, columns=[f'proba {label}' for label in LABEL_NAMES])
    test_df = test_df.reset_index(drop=True)
    test_df = pd.concat([test_df, proba_df], axis=1)
    test_df.to_csv('results/probas.csv', index=False)

    return test_df


def _generate_scores_csv(test_df):
    index = ['threshold',
             'upper midpoint',
             'lower midpoint']

    scores_df = pd.DataFrame(index=index, columns=LABEL_NAMES)

    scores_df.loc['threshold'] = [threshold for threshold in best_thresholds]

    for label, threshold in zip(LABEL_NAMES, best_thresholds):
        scores_df.loc['upper midpoint', label] = (threshold + 1) / 2
        scores_df.loc['lower midpoint', label] = threshold / 2
                                                                        
    scores_df.to_csv('results/scores.csv')


def _split_on_review_ids(train_df):
    review_ids = train_df['review_id'].unique().tolist()
    test_size = int(0.2 * len(review_ids))
    test_ids = random.sample(review_ids, test_size)
    test_df = train_df[train_df['review_id'].isin(test_ids)]
    review_ids = [id for id in review_ids if id not in test_ids]
    train_df = train_df[~train_df['review_id'].isin(test_ids)]
    return train_df, test_df


def _model_init():
    config = AutoConfig.from_pretrained(MODEL_CKPT, num_labels=len(LABEL_NAMES), problem_type='multi_label_classification')
    config.num_labels = len(LABEL_NAMES)
    config.id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
    config.label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, config=config).to(DEVICE)
    return model


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy()
    preds = (preds > best_thresholds).astype(int) # default threshold
    f1 = f1_score(labels, preds, average='samples')
    return {'f1': f1}


def _optuna_hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    }


def _validate(train_df):
    global best_thresholds
    
    train_df, val_df = _split_on_review_ids(train_df)

    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': train_df['original_sentence'], 'labels': train_df[LABEL_NAMES].values.astype(np.float32)}),
        'validation': Dataset.from_dict({'text': val_df['original_sentence'], 'labels': val_df[LABEL_NAMES].values.astype(np.float32)})
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

    save_path = 'models/hp_search/distilbert-finetuned'

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=50,
        evaluation_strategy='steps',
        disable_tqdm=False,
        logging_steps=100,
        log_level='error',
        load_best_model_at_end=True
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    best_thresholds = np.full(len(LABEL_NAMES), 0.5) # initialise global variable

    trainer = Trainer(
        model=None,
        model_init=_model_init,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=sentences_encoded['train'],
        eval_dataset=sentences_encoded['validation'],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    best_run = trainer.hyperparameter_search(
        direction='maximize',
        backend='optuna',
        hp_space=_optuna_hp_space,
        n_trials=10
    )

    model = _model_init()

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=50,
        learning_rate=best_run.hyperparameters['learning_rate'],
        per_device_train_batch_size=best_run.hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=best_run.hyperparameters['per_device_train_batch_size'],
        weight_decay=best_run.hyperparameters['weight_decay'],
        evaluation_strategy='steps',
        eval_steps=100,
        disable_tqdm=False,
        logging_steps=100,
        log_level='error',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=sentences_encoded['train'],
        eval_dataset=sentences_encoded['validation'],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    thresholds = np.arange(0.1, 0.9, 0.01)
    
    output = trainer.predict(sentences_encoded['validation']).predictions
    probas = torch.sigmoid(torch.from_numpy(output)).numpy()

    # # find best threshold for each label based on split validation set f1 score
    for i, label in enumerate(LABEL_NAMES):
        best_f1 = 0.0
        for threshold in thresholds:
            label_predictions = (probas[:, i] > threshold)
            f1 = f1_score(val_df[LABEL_NAMES].values[:, i], label_predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = threshold # update global variable

    print(f'Returning best run hyperparameters: {best_run.hyperparameters} and best thresholds: {best_thresholds}')

    return best_run.hyperparameters


def _train(train_df):
    train_df, val_df = _split_on_review_ids(train_df)

    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': train_df['original_sentence'], 'labels': train_df[LABEL_NAMES].values.astype(np.float32)}),
        'validation': Dataset.from_dict({'text': val_df['original_sentence'], 'labels': val_df[LABEL_NAMES].values.astype(np.float32)})
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

    best_hyperparameters = _validate(train_df)

    save_path = 'models/distilbert-finetuned'
    best_learning_rate = best_hyperparameters['learning_rate']
    best_batch_size = best_hyperparameters['per_device_train_batch_size']
    best_weight_decay = best_hyperparameters['weight_decay']

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=50,
        learning_rate=best_learning_rate,
        per_device_train_batch_size=best_batch_size,
        per_device_eval_batch_size=best_batch_size,
        weight_decay=best_weight_decay,
        evaluation_strategy='steps',
        eval_steps=100,
        disable_tqdm=False,
        logging_steps=100,
        log_level='error',
        load_best_model_at_end=True
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    model = _model_init()

    trainer = Trainer(
        model=None,
        model_init=_model_init,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=sentences_encoded['train'],
        eval_dataset=sentences_encoded['validation'],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    trainer.save_model('models/final')

    return trainer, tokenizer


def _predict(test_df, trainer, tokenizer):
    dataset = Dataset.from_dict({'text': test_df['original_sentence'], 'labels': test_df[LABEL_NAMES].values.astype(np.float32)})
    sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

    with torch.no_grad():
        pred_output = trainer.predict(sentences_encoded)

    print(f'pred_output.metrics (best thresholds): {pred_output.metrics}')

    probas = torch.sigmoid(torch.from_numpy(pred_output.predictions)).numpy()
    predictions = (probas > best_thresholds).astype(int)

    return probas, predictions


def _remove_checkpoints(dir_path):
    if os.path.isdir(dir_path):
        subdirs = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        for subdir in subdirs:
            try:
                shutil.rmtree(subdir)
            except OSError as e:
                print(f'Error: {subdir} : {e.strerror}')
        print(f'{dir_path} contents removed.')
    else:
        print(f'=====> {dir_path} not found. Remove manually.')


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/train.csv')
        df = df[df['cleaned_sentence'].notnull()]

        LABEL_NAMES = df.columns[5:].to_list()

        train_df, test_df = _split_on_review_ids(df)

        X_train = train_df[['review_id', 'original_sentence']].values
        y_train = train_df.iloc[:, 5:].values
        X_test = test_df[['review_id', 'original_sentence', 'cleaned_sentence']].values
        y_test = test_df.iloc[:, 5:].values

        # convert back to pandas dfs with label names
        train_df = pd.DataFrame({'review_id': X_train[:, 0], 'original_sentence': X_train[:, 1]})
        test_df = pd.DataFrame({'original_sentence': X_test[:, 1], 'cleaned_sentence': X_test[:, 2]})

        label_columns = df.columns[5:]

        for i, label_column in enumerate(label_columns):
            train_df[label_column] = y_train[:, i]
            test_df[label_column] = y_test[:, i]

        train_df = train_df.reset_index(drop=True)

        print('Training model...')
        trainer, tokenizer = _train(train_df)
        print('Done. Model trained.')

        print('Predicting...')
        probas, predictions = _predict(test_df, trainer, tokenizer)
        print('Done. Predictions made. Finetuned model saved in models/')

        for i, label in enumerate(LABEL_NAMES):
            test_df[f'pred {label}'] = predictions[:, i]
        test_df.to_csv('results/predictions.csv', index=False)

        _generate_cm_csv(test_df, predictions, y_test)
        print('Confusion matrices saved in results/cm/')
        test_df = _generate_probas_csv(test_df, probas)
        print('Probabilities saved in results/')

        _generate_scores_csv(test_df)
        print('Scores saved in results/')

        # _remove_checkpoints(os.path.abspath(f'C:\\Users\\{os.getlogin()}\\ray_results'))
        _remove_checkpoints('models/distilbert-finetuned')
        _remove_checkpoints('models/hp_search')
    except FileNotFoundError as e:
        print('data/train.csv not found.')


# TODO:
# - thresholds vary greatly between splits?
# - try different transformer models
#   https://www.kaggle.com/code/thedrcat/oversampling-for-multi-label-classification