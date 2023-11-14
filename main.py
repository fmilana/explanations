import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import remove_stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from augment import MLSMOTE, get_minority_samples
from vectorizer import Sentence2Vec
from sklearn.multioutput import ClassifierChain
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from eli5 import format_as_text, format_as_html
from eli5.lime import TextExplainer
from lime.lime_text import LimeTextExplainer
from xgboost import XGBClassifier


# from https://github.com/TeamHG-Memex/eli5/issues/337
class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, chains):
        self.chains = chains # XGBoost chains

    def fit(self, X, Y): # fit the XGBoost chains
        X, Y = self.oversample(X, Y) # oversample minority classes
        for i, chain in enumerate(self.chains): # fit each XGBoost chain
            chain.fit(X, Y)
            print(f"{i+1}/{len(self.chains)} chains fit")

    def predict(self, X): # get predictions from the XGBoost chains
        return np.rint(np.array([chain.predict(X) for chain in self.chains]).mean(axis=0)).astype(int) # predict with the XGBoost chains
    
    def predict_proba(self, X): # get prediction probas from the XGBoost chains
        if len(X) == 1:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)[0]
            sums_to = sum(self.probas_)
            new_probs = [x / sums_to for x in self.probas_]
            return new_probs
        else:
            self.probas_ = np.array([chain.predict_proba(X) for chain in self.chains]).mean(axis=0)
            print(self.probas_)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                # print(sums_to)
                new_probs = [x / sums_to for x in list_of_probs]
                ret_list.append(np.asarray(new_probs))
            return np.asarray(ret_list)
    
    def oversample(self, X, Y):
        X_shape_old = X.shape
        class_dist = [y/Y.shape[0] for y in Y.sum(axis=0)]
        print("checking for minority classes in train split...")
        X_sub, Y_sub = get_minority_samples(pd.DataFrame(X), pd.DataFrame(Y))

        # print(f"X_sub: {X_sub}")
        # print(f"Y_sub: {Y_sub}")

        if np.shape(X_sub)[0] > 0: # only oversample training set if minority samples are found
            print("minority classes found.")
            print("oversampling...")
            try:
                X_res, Y_res = MLSMOTE(X_sub, Y_sub, round(X.shape[0]/5))       
                X = np.concatenate((X, X_res.to_numpy())) # append augmented samples
                Y = np.concatenate((Y, Y_res.to_numpy())) # to original dataframes
                print("oversampled.")
                class_dist_os = [y/Y.shape[0] for y in Y.sum(axis=0)]
                print("CLASS DISTRIBUTION:")
                print(f"Before MLSMOTE: {X_shape_old}, {class_dist}")
                print(f"After MLSMOTE: {X.shape}, {class_dist_os}")
            except ValueError:
                print("could not oversample because n_samples < n_neighbors in some classes")
        else:
            print("no minority classes.")
        return X, Y


def explain_pred(text_explainer, pipeline, categories, sentence):
    text_explainer.fit(sentence, pipeline.predict_proba)
    prediction = text_explainer.explain_prediction(target_names=categories)
    txt = format_as_text(prediction)
    txt_file = open("results/explanations.txt", "a+")
    txt_file.write(txt)
    txt_file.close()
    html = format_as_html(prediction)
    html_file = open("results/explanations.html", "a+")
    html_file.write(html)
    html_file.close()
    print(text_explainer.metrics_)


train_df = pd.read_csv("data/train.csv")

X_train = np.array(train_df["sentence_embedding"].tolist())
Y_train = np.array(train_df.iloc[:, 2:])

clf = XGBClassifier()
number_of_chains = 10
chains = [ClassifierChain(clf, order="random", random_state=i) for i in range(number_of_chains)]

model = MultiLabelProbClassifier(chains)

pipeline = make_pipeline(Sentence2Vec(), model) 
pipeline.fit(X_train, Y_train)


categories = ["food and drinks", "place", "people", "opinions"]

# FOOD AND DRINKS
# sentence = "They will even make you a burger in which the bun has been substituted for two halves of an avocado."
# sentence = "But the word-spaghetti of the menu descriptions – 'aubergine, kalamatas, tahini' reads one; \
# 'grilled beetroot, spring onion, smoked soy' reads another – is so alluring, so well lubricated with promise, that I give myself to it happily."

# PLACE
# sentence = "To the right, in a small parade, there's Neat Burger, knocking out pea protein and corn-based patties, \
#     dyed what they think are the right colours by the addition of beetroot and turmeric."
# sentence = "Now, the upstairs dining room has parquet floors, comfortable midcentury modern tan leather chairs and a kitchen with principles."
# sentence = "But the most interesting of these three restaurants on Princes Street, tucked in together for comfort, is in the middle."
sentence = "Tendril started as a pop-up, first in a Soho pub, then later here, on this narrow site just south of Oxford Street."

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

# clear explanations.txt and explanations.html
open('results/explanations.txt', 'w').close()
open('results/explanations.html', 'w').close()
print("cleared explanations.txt and explanations.html")

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

# eli5
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