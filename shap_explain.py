import shap
import numpy as np
from pathlib import Path


# https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras
def get_shap_weights(pipeline, class_names, sentence, class_name):
    explainer = shap.Explainer(pipeline.predict, masker=shap.maskers.Text(tokenizer=r'\b\w+\b'), output_names=class_names)

    explanation = explainer([sentence])

    shap_array = np.squeeze(explanation.values)

    shap_weights = shap_array[:, class_names.index(class_name)].tolist()

    print(f'=================> {len(shap_weights)} shap_weights: {shap_weights}')

    return shap_weights