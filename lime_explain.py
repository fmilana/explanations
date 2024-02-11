import re
from pathlib import Path
from eli5 import format_as_text, format_as_html, format_as_dict
from eli5.lime import TextExplainer


txt_path = Path("results/lime/lime.txt")
html_path = Path("results/lime/lime.html")


def save_to_files(explanation):
    txt = format_as_text(explanation)
    html = format_as_html(explanation)
    
    with open(txt_path, "a+", encoding="utf-8") as f:
        f.write(txt)
    print(f"saved to {txt_path}")
    
    with open(html_path, "a+", encoding="utf-8") as f:
        f.write(html)
    print(f"saved to {html_path}")


def run_lime(pipeline, categories, sentence):
    # clear txt and html
    if Path(txt_path):
        with open(txt_path, "w", encoding="utf-8") as f:
            pass
        print(f"cleared {txt_path}")
    if Path(html_path):
        with open(html_path, "w", encoding="utf-8") as f:
            pass
        print(f"cleared {html_path}")

    n_samples_list = [300, 1000, 2000, 3000, 4000, 5000]

    best_score = 0
    best_dict = {}

    for n_samples in n_samples_list:
        text_explainer = TextExplainer(
            token_pattern=r"\b\w+\b",
            n_samples=n_samples,
            position_dependent=True,
            random_state=42
        )

        print(f"fitting lime text_explainer with n_samples={n_samples}")        

        text_explainer.fit(sentence, pipeline.predict_proba)

        explanation = text_explainer.explain_prediction(target_names=categories)

        save_to_files(explanation)

        pred_dict = format_as_dict(explanation)

        metrics = text_explainer.metrics_

        print(metrics)

        score = metrics["score"]
        
        if score > best_score:
            best_score = score
            best_dict = pred_dict

    return best_dict


def get_lime_weights(pipeline, class_names, sentence, class_name):
    lime_dict = run_lime(pipeline, class_names, sentence)

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

    print(f"lime_tuples: {lime_tuples}")

    lime_bias = lime_weights.pop(0)

    return lime_bias, lime_weights