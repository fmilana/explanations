from pathlib import Path


html_path = Path("results/html/results.html")


def create_html(sentence, lime_bias, lime_values, shap_values):
    return