import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from classifier import MultiLabelProbClassifier
from vectorizer import Sentence2Vec


train_df = pd.read_csv("data/train.csv")

X_train = train_df["original_sentence"].tolist()
Y_train = np.array(train_df.iloc[:, 3:])

pipeline = make_pipeline(Sentence2Vec(), MultiLabelProbClassifier())

pipeline.fit(X_train, Y_train)

categories = ["food and drinks", "place", "people", "opinions"]

# sentence = "To cook their steaks, they built a huge charcoal grill into the open kitchen"
# sentence = "It is an elbows-on-the-table sort of place, where you could hunker down by yourself over a bowl of something steaming and delicious that makes the world better, and all at an extremely good price."
# sentence = "They will even make you the boneless inbetween which another bun has been refered without two halves of another swede"
# sentence = "They will even make you thievery burger in which the bun has been substituted for two halves among an fruit ."

# prediction = pipeline.predict([sentence]).flatten()
# print(f"prediction: {prediction}")
# try:
#     predicted_category = categories[np.where(prediction==1)[0][0]]
# except IndexError:
#     predicted_category = "None"
# predict_proba = pipeline.predict_proba([sentence]).flatten()

# print(f"predicted_category: \"{predicted_category}\"")
# print(f"predict_proba: {predict_proba}")


sentences = [
    "To cook their steaks, they built a huge charcoal grill into the open kitchen",
    "It is an elbows-on-the-table sort of place, where you could hunker down by yourself over a bowl of something steaming and delicious that makes the world better, and all at an extremely good price.",
    "From Thursday to Saturday, they offer a five-course tasting menu at £40 a head.",
    "I would come back here just for this, and not quite twitch at the £25 price tag."
]

for sentence in sentences:
    predict_probas = pipeline.predict_proba([sentence]).flatten()
    print(f"prediction: {predict_probas}")