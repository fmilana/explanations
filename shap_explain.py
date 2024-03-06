import shap
import numpy as np


# https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras
def get_shap_weights(pipeline, class_names, sentence, class_name):
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')

    # predict_proba_shap is a function of CustomPipeline in custom_pipeline.py
    explainer = shap.Explainer(pipeline.predict_proba_for_shap, masker=masker, output_names=class_names)

    explanation = explainer([sentence])

    shap_2d_array = np.squeeze(explanation.values)

    shap_weights = shap_2d_array[:, class_names.index(class_name)].tolist()

    return shap_weights