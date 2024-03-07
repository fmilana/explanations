import random
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score


BEST_THRESHOLDS = 0.5


def _compute_metrics(pred):
    global BEST_THRESHOLDS
    labels = pred.label_ids
    preds = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy()
    preds = (preds > BEST_THRESHOLDS).astype(int)
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}


def _train(model, sentences_encoded_train, sentences_encoded_val, tokenizer):
    batch_size = 64
    logging_steps = len(sentences_encoded_train) // batch_size
    save_path = 'model/distilbert-finetuned'
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=50,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy='steps',
        disable_tqdm=False,
        logging_steps=logging_steps,
        save_steps=logging_steps,
        log_level='error',
        load_best_model_at_end=True,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=sentences_encoded_train,
        eval_dataset=sentences_encoded_val,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    prediction_output = trainer.predict(sentences_encoded_val)
    probas = torch.sigmoid(torch.from_numpy(prediction_output.predictions)).numpy()

    return probas


def _cross_validate(df, tokenizer):
    global NUM_LABELS
    review_ids = df['review_id'].unique().tolist()
    test_size = 5 # arbitrary
    test_ids = random.sample(review_ids, test_size)
    test_df = df[df['review_id'].isin(test_ids)]
    review_ids = [id for id in review_ids if id not in test_ids]
    df = df[~df['review_id'].isin(test_ids)]

    thresholds = np.arange(0.1, 0.9, 0.1)

    best_thresholds_per_label = np.zeros((len(review_ids), NUM_LABELS))
    best_scores_per_label = np.zeros((len(review_ids), NUM_LABELS))

    for iteration_index, review_id in enumerate(review_ids):
        print(f'Iteration {iteration_index + 1}/{len(review_ids)}')
        validation_df = df[df['review_id'] == review_id]
        train_df = df[~df['review_id'].isin(test_ids + [review_id])]

        X_train = np.array(train_df['original_sentence'].to_list())
        Y_train = np.array(train_df.iloc[:, -NUM_LABELS:])

        X_val = np.array(validation_df['original_sentence'].to_list())
        y_val = np.array(validation_df.iloc[:, -NUM_LABELS:])

        dataset = DatasetDict({
            'train': Dataset.from_dict({'text': X_train, 'labels': Y_train.astype(np.float32)}),
            'validation': Dataset.from_dict({'text': X_val, 'labels': y_val.astype(np.float32)})
        })

        sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

        # Reinitialize the model for each split
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS, problem_type='multi_label_classification').to(device)

        probas = _train(model, sentences_encoded['train'], sentences_encoded['validation'], tokenizer)

        for label_index in range(NUM_LABELS):
            y_class_true = y_val[:, label_index]
            best_score_for_label = 0
            best_threshold_for_label = 0
            for threshold in thresholds:
                y_class_pred = (probas[:, label_index] > threshold).astype(int)
                score = f1_score(y_class_true, y_class_pred, average='weighted')
                if score > best_score_for_label:
                    best_score_for_label = score
                    best_threshold_for_label = threshold
        
            best_thresholds_per_label[iteration_index, label_index] = best_threshold_for_label
            best_scores_per_label[iteration_index, label_index] = best_score_for_label

    average_best_thresholds_per_label = np.mean(best_thresholds_per_label, axis=0)
    average_best_scores_per_label = np.mean(best_scores_per_label, axis=0)

    print(f'===============VALIDATION RESULTS===============')
    print(f'average best thresholds per class: {average_best_thresholds_per_label}')
    print(f'average scores per class: {average_best_scores_per_label}')
    print(f'average score across all classes: {np.mean(average_best_scores_per_label)}')
    print(f'================================================')

    return average_best_thresholds_per_label


def _train_and_validate(df):
    global NUM_LABELS
    global BEST_THRESHOLDS
    label_names = df.columns[7:].to_list()
    NUM_LABELS = len(label_names)
    model_ckpt = 'distilbert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=NUM_LABELS, problem_type='multi_label_classification').to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    BEST_THRESHOLDS = _cross_validate(df, tokenizer)

    X_train, y_train, X_val, y_val = iterative_train_test_split(df[['original_sentence']].values, df[label_names].values, test_size = 0.2)

    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': X_train[:, 0], 'labels': y_train.astype(np.float32)}),
        'validation': Dataset.from_dict({'text': X_val[:, 0], 'labels': y_val.astype(np.float32)})
    })
    
    sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

    _train(model, sentences_encoded['train'], sentences_encoded['validation'], tokenizer)





if __name__ == '__main__':
    try:
        df = pd.read_csv('data/train.csv')
        df = df[df['cleaned_sentence'].notnull()]
        print('Training and validating model...')
        fine_tuned_model = _train_and_validate(df)
        # save the model to disk
        
        print('Done. Finetuned model saved in model/')
    except FileNotFoundError as e:
        print('data/train.csv not found.')
