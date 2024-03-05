from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import torch


def tokenize(batch):
    return tokenizer(batch['original_sentence'], padding=True, truncation=True)


def extract_hidden_state(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {'hidden_state': last_hidden_state[:, 0].cpu().numpy()}


model_ckpt = 'bert-base-uncased'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

df = pd.read_csv('data/train.csv')

sentences_df = df['original_sentence'].to_frame()

dataset = Dataset.from_pandas(sentences_df)

sentences_encoded = dataset.map(tokenize, batched=True, batch_size=None)

sentences_encoded.set_format('torch', columns=['input_ids'])

sentences_hidden = sentences_encoded.map(extract_hidden_state, batched=True)

X_train = np.array(sentences_hidden['hidden_state'])

sentence_embeddings = [str(embedding) for embedding in X_train.tolist()]

df['sentence_embedding'] = sentence_embeddings

df.to_csv('data/train_bert.csv', index=False)