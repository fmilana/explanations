import pandas as pd
import numpy as np
from pathlib import Path
from classifier import MultiLabelProbClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import remove_stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from vectorizer import Sentence2Vec
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import make_pipeline
from eli5 import format_as_text, format_as_html
from eli5.lime import TextExplainer
# from lime.lime_text import LimeTextExplainer


txt_path = Path("results/lime/lime.txt")
html_path = Path("results/lime/lime.html")


def explain_pred(text_explainer, pipeline, categories, sentence):
    text_explainer.fit(sentence, pipeline.predict_proba)
    prediction = text_explainer.explain_prediction(target_names=categories)
    txt = format_as_text(prediction)
    with open(txt_path, "a+") as txt_file:
        txt_file.write(txt)
    print(f"saved to {txt_path}")
    html = format_as_html(prediction)
    with open(html_path, "a+") as html_file:
        html_file.write(html)
    print(f"saved to {html_path}")
    print(text_explainer.metrics_)


train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 2:])

print("calling make_pipeline")
pipeline = make_pipeline(Sentence2Vec(), MultiLabelProbClassifier())
print("done calling make_pipeline")
print("calling pipeline.fit")
pipeline.fit(X_train, Y_train)
print("done calling pipeline.fit")


categories = ["food and drinks", "place", "people", "opinions"]

# FOOD AND DRINKS
# sentence = "They will even make you a burger in which the bun has been substituted for two halves of an avocado."
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

# food and drinks false positives
sentence = "Drink orders are forgotten, then have to be re-explained."

# food and drinks false negatives
sentence = "It's a tough, extremely indifferent piece of meat."

prediction = pipeline.predict([sentence]).flatten()
try:
    predicted_category = categories[np.where(prediction==1)[0][0]]
except IndexError:
    predicted_category = "None"
predict_proba = pipeline.predict_proba([sentence]).flatten()

print(f"predicted_category: \"{predicted_category}\"")
print(f"predict_proba: {predict_proba}")

# clear lime.txt and lime.html
if Path(txt_path):
    open(txt_path, "w").close()
    print(f"cleared {txt_path}")
if Path(html_path):
    open(html_path, "w").close()
    print(f"cleared {html_path}")

n_samples_list = [
    300,
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    15000,
    20000,
    25000,
    30000
    ]

# eli5 (lime)
for n_samples in n_samples_list:
    text_explainer = TextExplainer(
        # clf=DecisionTreeClassifier(max_depth=n_samples),
        # clf=RandomForestClassifier(max_depth=None, n_estimators=100, max_features=None, random_state=42),
        n_samples=n_samples,
        position_dependent=True,
        random_state=42
    )

    print(f"calling explain_pred with n_samples={n_samples}")
    explain_pred(text_explainer, pipeline, categories, sentence)


# lime
# https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html
# print("Chains predictions:")
# print(pipeline.predict_proba([sentence]))
# print("=====================================")

# explainer = LimeTextExplainer(class_names=categories)
# print(f'len(sentence.split())={len(sentence.split())}')
# exp = explainer.explain_instance(
#     sentence, pipeline.predict_proba, 
#     num_features=len(sentence.split()), 
#     labels=[i for i, _ in enumerate(categories)]
#     )

# for i in range(len(categories)):
#     print(f"Explanation for class {categories[i]}:")
#     print("\n".join(map(str, exp.as_list(label=i))))
#     print()

# exp.save_to_file('results/lime.html')
# debug: https://github.com/marcotcr/lime/issues/243