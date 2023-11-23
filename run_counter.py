from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = "png"
from omnixai.data.text import Text
from omnixai.explainers.nlp.counterfactual.polyjuice import Polyjuice
from sklearn.pipeline import make_pipeline
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec

# install polyjuice manually (might require pkg-config + mysqlclient first)
# follow this guide to fix polyjuice: https://github.com/tongshuangwu/polyjuice/issues/12

# https://opensource.salesforce.com/OmniXAI/latest/tutorials/nlp/ce_classification.html
txt_path = Path("results/counter/counter.txt")

train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 2:])

pipeline = make_pipeline(Sentence2Vec(), MultiLabelProbClassifier())

pipeline.fit(X_train, Y_train)

categories = ["food and drinks", "place", "people", "opinions"]

sentence = "They will even make you a burger in which the bun has been substituted for two halves of an avocado."
# sentence = "But the word-spaghetti of the menu descriptions – 'aubergine, kalamatas, tahini' reads one; \
# 'grilled beetroot, spring onion, smoked soy' reads another – is so alluring, so well lubricated with promise, that I give myself to it happily."

# PLACE
# sentence = "To the right, in a small parade, there's Neat Burger, knocking out pea protein and corn-based patties, \
# #     dyed what they think are the right colours by the addition of beetroot and turmeric."
# sentence = "Now, the upstairs dining room has parquet floors, comfortable midcentury modern tan leather chairs and a kitchen with principles."
# sentence = "But the most interesting of these three restaurants on Princes Street, tucked in together for comfort, is in the middle."
# sentence = "Tendril started as a pop-up, first in a Soho pub, then later here, on this narrow site just south of Oxford Street."

# PEOPLE
# sentence = "Head chef Graham Chatham, who has cooked at Rules and Daylesford Organic, treats them with old school care, attention and at times, maternal indulgence."
# sentence = "The service is sometimes chaotic but, like a primary school ballet class, always enthusiastic."
# sentence = "That's exactly what you would expect of a chef like Shaun Moffat, who has cooked in London at the \
# Middle Eastern-inflected Berber & Q, and at the cheerfully iconoclastic Manteca, which treats the Italian \
# classics as a mere opening position in a ribald negotiation."

# OPINIONS
# sentence = "There's an awful lot going on here."
# sentence = "It's restless but focused and jolly."
# sentence = "Just go elsewhere afterwards for an ice-cream."
# sentence = "Importantly though, it is good value."

predict_proba = pipeline.predict_proba([sentence]).flatten()
prediction = pipeline.predict([sentence]).flatten()

print(f"predict_proba: {predict_proba}")
print(f"prediction: {prediction}")

explainer = Polyjuice(predict_function=pipeline.predict_proba)

explanations = explainer.explain(Text([sentence])) # omnixai.explanations.tabular.counterfactual.CFExplanation
# print(type(explanations))
# print(dir(explanations))
# fig = plt.figure() 
# explanations.plot()
# plt.show()

explanations_list = explanations.get_explanations()
explanations_dict = explanations_list[0] # get explanations for first sentence

if Path(txt_path):
    open(txt_path, "w").close()
    print(f"cleared {txt_path}")

with open(txt_path, "r+") as f:
    for key, df in explanations_dict.items():
        f.write(f"{key}:\n")
        for index, row in df.iterrows():
            category = categories[int(row["label"])]
            text = row["text"]
            f.write(f"- ({category}) {text}\n")
        f.write("\n")
    f.writelines(f.readlines()[:-2]) # remove last two newlines (to-do: fix)
print(f"saved to {txt_path}")