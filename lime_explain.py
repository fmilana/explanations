import re
from eli5 import format_as_dict
from eli5.lime import TextExplainer


def run_lime(pipeline, categories, sentence, optimized):
    if optimized:
        n_samples_list = [300, 1000, 2000, 3000, 4000, 5000]
    else:
        n_samples_list = [2000]

    best_score = 0
    best_dict = {}

    for n_samples in n_samples_list:
        text_explainer = TextExplainer(token_pattern=r"\b\w+\b", n_samples=n_samples, position_dependent=True, random_state=42)

        print(f"fitting lime text_explainer with n_samples={n_samples}")        

        text_explainer.fit(sentence, pipeline.predict_proba)

        explanation = text_explainer.explain_prediction(target_names=categories)

        pred_dict = format_as_dict(explanation)

        metrics = text_explainer.metrics_

        print(f"lime metrics: {metrics}")

        score = metrics["score"]
        
        if score > best_score:
            best_score = score
            best_dict = pred_dict

    return best_dict


def get_lime_weights(pipeline, class_names, sentence, class_name, optimized):
    lime_dict = run_lime(pipeline, class_names, sentence, optimized)

    target = lime_dict["targets"][class_names.index(class_name)]
    positive_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["pos"]]
    negative_weight_tuples = [(entry["feature"], entry["weight"]) for entry in target["feature_weights"]["neg"]]

    lime_tuples = positive_weight_tuples + negative_weight_tuples
    
    # add missing words (words are missing if their LIME weight is 0.0)
    all_words = set(f"[{i}] {word}" for i, word in enumerate(re.findall(r"\b\w+\b", sentence)))
    lime_words = set(word for word, weight in lime_tuples)
    missing_words = all_words - lime_words
    lime_tuples.extend((word, 0.0) for word in missing_words)

    lime_tuples = sorted(lime_tuples, key=lambda x: int(re.search(r'\[(\d+)\]', x[0]).group(1)) if re.search(r'\[(\d+)\]', x[0]) is not None else -1)

    lime_weights = [tuple[1] for tuple in lime_tuples]

    lime_bias = lime_weights.pop(0)

    print(f"=================> {len(lime_weights)} lime_weights: {lime_weights}")

    return lime_bias, lime_weights