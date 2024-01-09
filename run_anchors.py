from pathlib import Path
import spacy
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from alibi.utils import spacy_model
from alibi.explainers import AnchorText
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec
from preprocess import remove_stop_words


# https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_text_movie.html
txt_path = Path("results/anchors/anchors.txt")

train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 2:])

pipeline = make_pipeline(Sentence2Vec(), MultiLabelProbClassifier())

pipeline.fit(X_train, Y_train)

categories = ["food and drinks", "place", "people", "opinions"]

sentence = "They will even make you a burger in which the bun has been substituted for two halves of an avocado."
# sentence = "Tendril started as a pop-up, first in a Soho pub, then later here, on this narrow site just south of Oxford Street.",
# sentence = "Head chef Graham Chatham, who has cooked at Rules and Daylesford Organic, treats them with old school care, attention and at times, maternal indulgence.",
# sentence = "The service is sometimes chaotic but, like a primary school ballet class, always enthusiastic.",
# sentence = "That's exactly what you would expect of a chef like Shaun Moffat, who has cooked in London at the \
#     Middle Eastern-inflected Berber & Q, and at the cheerfully iconoclastic Manteca, which treats the Italian \
#     classics as a mere opening position in a ribald negotiation."

sentence = remove_stop_words(sentence)

# prediction = pipeline.predict([sentence]).flatten()
prediction = pipeline.predict([sentence])
try:
    predicted_category = categories[np.where(prediction==1)[0][0]]
except IndexError:
    predicted_category = "None"
predict_proba = pipeline.predict_proba([sentence]).flatten()

print(f"predicted_category: \"{predicted_category}\"")
print(f"predict_proba: {predict_proba}")

print('loading spaCy model...')
model = "en_core_web_md"
spacy_model(model=model)

# https://docs.seldon.io/projects/alibi/en/latest/examples/anchor_text_movie.html
explainer = AnchorText(
    predictor=pipeline.predict,
    sampling_strategy="similarity",
    nlp=spacy.load(model),
    use_proba=True, # sample according to the similiary distribution
    sample_proba=0.5, # probability of a word to be masked and replaced by a similar word
    top_n=20, # consider only top 20 words most similar words
    temperature=0.2 # higher temperature implies more randomness when sampling
)

explanation = explainer.explain(sentence, threshold=0.95)

if Path(txt_path):
    open(txt_path, "w").close()
    print(f"cleared {txt_path}")

with open(txt_path, "w") as f:
    f.write("Original sentence: %s" % sentence)
    f.write("\nPrediction: %s" % predicted_category)
    f.write("\n\nAnchor: %s" % (" AND ".join(explanation.anchor)))
    f.write("\nPrecision: %.2f" % explanation.precision)
    f.write(f"\n\nExamples where anchor applies and model predicts \"{predicted_category}\":\n- ")
    f.write("\n- ".join([x for x in explanation.raw["examples"][-1]["covered_true"]]))
    f.write(f"\n\nExamples where anchor applies and model does NOT predict \"{predicted_category}\":\n- ")
    f.write("\n- ".join([x for x in explanation.raw["examples"][-1]["covered_false"]]))

print(f"saved to {txt_path}")