import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score


NUM_CLASSES = 0


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy() > 0.5
    # preds = (preds > BEST_THRESHOLDS).astype(int) # todo:use best thresholds instead
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}


def _train_and_validate(df):
    global NUM_CLASSES
    label_names = df.columns[7:].to_list()
    NUM_CLASSES = len(label_names)
    model_ckpt = 'distilbert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=NUM_CLASSES, problem_type='multi_label_classification').to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    X_train, y_train, X_val, y_val = iterative_train_test_split(df[['original_sentence']].values, df[label_names].values, test_size = 0.2)

    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': X_train[:, 0], 'labels': y_train.astype(np.float32)}),
        'validation': Dataset.from_dict({'text': X_val[:, 0], 'labels': y_val.astype(np.float32)})
    })
    
    sentences_encoded = dataset.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=None)

    batch_size = 64
    logging_steps = len(sentences_encoded['train']) // batch_size
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
        train_dataset=sentences_encoded['train'],
        eval_dataset=sentences_encoded['validation'],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    trainer.train()





if __name__ == '__main__':
    try:
        df = pd.read_csv('data/train.csv')
        print('Training and validating model...')
        fine_tuned_model = _train_and_validate(df)
        # save the model to disk
        
        print('Done. Finetuned model saved in model/')
    except FileNotFoundError as e:
        print('data/train.csv not found.')
