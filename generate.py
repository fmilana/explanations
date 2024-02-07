import re
from draw import add_title_to_html, add_to_html
from run_lime import generate_lime
from run_shap import generate_shap
from sklearn.pipeline import make_pipeline
from vectorizer import Sentence2Vec


def generate_lime_weights(pipeline, class_names, sentence, predicted_class):
    lime_dict = generate_lime(pipeline, class_names, sentence)

    target = lime_dict["targets"][class_names.index(predicted_class)]
    positive_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["pos"]]
    negative_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["neg"]]

    lime_tuples = positive_weight_tuples + negative_weight_tuples
    lime_tuples = sorted(lime_tuples, key=lambda x: int(re.search(r'\[(\d+)\]', x[0]).group(1)) if re.search(r'\[(\d+)\]', x[0]) is not None else -1)

    lime_weights = [tuple[1] for tuple in lime_tuples]

    lime_bias = lime_weights.pop(0)

    return lime_bias, lime_weights


def generate_shap_weights(pipeline, class_names, sentence, class_name):
    shap_array = generate_shap(pipeline, class_names, sentence)
    shap_weights = shap_array[:, class_names.index(class_name)].tolist()

    return shap_weights


def generate_html(clf, sentence_dict):
    # clear html
    open("results/html/results.html", "w").close()

    pipeline = make_pipeline(Sentence2Vec(), clf)
    
    class_names = list(sentence_dict.keys())

    for class_name in sentence_dict.keys():
        top_positive_query_tuple = sentence_dict[class_name][0]
        q1_positive_query_tuple = sentence_dict[class_name][1]
        q3_negative_query_tuple = sentence_dict[class_name][2]
        bottom_negative_query_tuple = sentence_dict[class_name][3]
        tp_examples_tuples = sentence_dict[class_name][4]
        fp_examples_tuples = sentence_dict[class_name][5]
        fn_examples_tuples = sentence_dict[class_name][6]

        titles = ['True Positives', 'False Positives', 'False Negatives']

        for i, list_of_tuples in enumerate([tp_examples_tuples, fp_examples_tuples, fn_examples_tuples]):
            add_title_to_html(f'{class_name} {titles[i]}')

            for (sentence, cleaned_sentence, proba) in list_of_tuples:
                lime_bias, lime_weights = generate_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
                shap_weights = generate_shap_weights(pipeline, class_names, cleaned_sentence, class_name)

                add_to_html(sentence, proba, lime_bias, lime_weights, shap_weights)
                
        titles = ['Top Positive', 'Q1 Positive', 'Q3 Negative', 'Bottom Negative']        

        for i, (sentence, cleaned_sentence, proba) in enumerate([top_positive_query_tuple, q1_positive_query_tuple, q3_negative_query_tuple, bottom_negative_query_tuple]):
            add_title_to_html(f'{class_name} {titles[i]} Query')

            lime_bias, lime_weights = generate_lime_weights(pipeline, class_names, cleaned_sentence, class_name)
            shap_weights = generate_shap_weights(pipeline, class_names, cleaned_sentence, class_name)

            add_to_html(sentence, proba, lime_bias, lime_weights, shap_weights)

        
    # train_df = pd.read_csv("data/train.csv")

    # X_train = train_df["original_sentence"].tolist()
    # Y_train = np.array(train_df.iloc[:, 7:])

    # print("calling make_pipeline")
    # pipeline = make_pipeline(Sentence2Vec())
    # print("done calling make_pipeline")
    # print("calling pipeline.fit")
    # pipeline.fit(X_train, Y_train)
    # print("done calling pipeline.fit")

    # class_names = sentence_dict.keys()

    # sentence = "This is not cooking that redefines the very notion of Greek food."
    # # sentence = "Head chef Graham Chatham, who has cooked at Rules and Daylesford Organic, treats them with old school care, attention and at times, maternal indulgence."
    # cleaned_sentence = remove_stop_words(sentence)

    # prediction = pipeline.predict([cleaned_sentence]).flatten()
    # try:
    #     class_name = class_names[np.where(prediction==1)[0][0]]
    # except IndexError:
    #     class_name = "None"
    # predict_proba = pipeline.predict_proba([cleaned_sentence]).flatten()

    # class_name_proba = predict_proba[class_names.index(class_name)]

    # print(f"class_name: \"{class_name}\"")
    # print(f"predict_proba: {predict_proba}")

    # lime_bias, lime_weights = generate_lime_weights(pipeline, class_names, cleaned_sentence)
    # shap_weights = generate_shap_weights(pipeline, class_names, cleaned_sentence)

    # create_html(
    #     sentence, 
    #     class_name, 
    #     class_name_proba, 
    #     lime_bias, 
    #     lime_weights, 
    #     shap_weights
    #     )