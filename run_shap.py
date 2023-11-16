import re
import shap
import numpy as np
import pandas as pd
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import make_pipeline


def custom_tokenizer(s, return_offsets_mapping=True):
        """Custom tokenizers conform to a subset of the transformers API."""
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(r"\W", s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])
        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out

train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 2:])

clf = XGBClassifier()
number_of_chains = 10
chains = [ClassifierChain(clf, order="random", random_state=i) for i in range(number_of_chains)]

model = MultiLabelProbClassifier(chains)

pipeline = make_pipeline(Sentence2Vec(), model)

pipeline.fit(X_train, Y_train)

explainer = shap.Explainer(pipeline.predict, custom_tokenizer)

sentence = "They will even make you a burger in which the bun has been substituted for two halves of an avocado.",
# sentence = "Tendril started as a pop-up, first in a Soho pub, then later here, on this narrow site just south of Oxford Street.",
# sentence = "Head chef Graham Chatham, who has cooked at Rules and Daylesford Organic, treats them with old school care, attention and at times, maternal indulgence.",
# sentence = "The service is sometimes chaotic but, like a primary school ballet class, always enthusiastic.",
# sentence = "That's exactly what you would expect of a chef like Shaun Moffat, who has cooked in London at the \
#     Middle Eastern-inflected Berber & Q, and at the cheerfully iconoclastic Manteca, which treats the Italian \
#     classics as a mere opening position in a ribald negotiation."

print(f"pipeline.predict(sentences) = {pipeline.predict([sentence])}")

shap_values = explainer([sentence])

shap.plots.text(shap_values)