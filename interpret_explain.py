import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSequenceClassification.from_pretrained('models/final').to(device)
tokenizer = AutoTokenizer.from_pretrained('models/final')

cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)

sentence = "A slab of roasted turbot for £48 is as good as it should be at that price and comes with a solid romesco sauce"

word_attributions = cls_explainer(sentence)

cls_explainer.visualize('results/html/interpret_explain.html')