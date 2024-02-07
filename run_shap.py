import shap
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from augment import oversample
from classifier import MultiLabelProbClassifier
from preprocess import remove_stop_words
from vectorizer import Sentence2Vec
from sklearn.pipeline import make_pipeline

# https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras

html_path = Path("results/shap/shap.html")
png_path = Path("results/shap/shap.png")


def generate_shap(pipeline, categories, sentence):
    explainer = shap.Explainer(pipeline.predict,
                            masker=shap.maskers.Text(tokenizer=r"\W+"),
                            output_names=categories)

    explanation = explainer([sentence])

    squeezed_values = np.squeeze(explanation.values)
    print(f"type(explanation) : {type(explanation)}")
    print(f"SHAP Squeezed Values Shape : {squeezed_values.shape}")
    print(f"SHAP Base Values  : {explanation.base_values}")
    print(f"SHAP Data: {explanation.data[0]}")
    print(f"type SHAP Values : {type(squeezed_values)}")
    print(f"SHAP Values = {squeezed_values}")

    if Path(html_path):
        open(html_path, "w").close()
        print(f"cleared {html_path}")

    with open(html_path, "a+") as html_file:
        html_file.write(shap.plots.text(explanation, display=False))
    print(f"saved to {html_path}")

    # ----------------png plot giving index error-----------------
    # fig, axs = plt.subplots(1, len(categories), layout="constrained")

    # for i in range(len(categories)):
    #     print(f'Plotting {categories[i]}...')
        
    #     plt.sca(axs[i])
    #     axs[i].set_title(categories[i])
    #     # plt.figure(figsize=(20,20))
    #     shap.plots.bar(explanation[:, :, categories[i]].mean(axis=0),
    #                     max_display=len(sentence.split()),
    #                     order=shap.Explanation.argsort.flip,
    #                     # order=shap.Explanation.abs,
    #                     show=False)

    # fig.set_size_inches(6*len(categories), 10)
    # # plt.show()
    # plt.savefig(png_path)
    # print(f'saved to {png_path}')

    return np.squeeze(explanation.values)


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")

    X_train = train_df["original_sentence"].tolist()
    Y_train = np.array(train_df.iloc[:, 7:])

    X_train, Y_train = oversample(X_train, Y_train)

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

    prediction = pipeline.predict([sentence]).flatten()
    try:
        predicted_category = categories[np.where(prediction==1)[0][0]]
    except IndexError:
        predicted_category = "None"
    predict_proba = pipeline.predict_proba([sentence]).flatten()

    print(f"predicted_category: \"{predicted_category}\"")
    print(f"predict_proba: {predict_proba}")

    generate_shap(pipeline, categories, sentence)