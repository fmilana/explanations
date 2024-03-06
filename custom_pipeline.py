from sklearn.pipeline import Pipeline


class CustomPipeline(Pipeline):
    # get predict_proba for SHAP (which requires probas to be list (or array?) as input, even when there's only one sample)
    def predict_proba_for_shap(self, X):
        X = X.tolist() # convert to list (shap explainer converts to np array?)
        probas = super().predict_proba(X)
        if isinstance(probas, list):
            return [probas]
        return probas
