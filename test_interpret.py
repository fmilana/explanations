import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSequenceClassification.from_pretrained('models/final').to(device)
tokenizer = AutoTokenizer.from_pretrained('models/final')

cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)

sentence = "the wide rim of the steel fitting into which the cauldron fits contains curved metal compartments "

word_attributions = cls_explainer(sentence)

cls_explainer.visualize('results/html/interpret_explain.html')