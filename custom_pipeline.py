from sklearn.pipeline import Pipeline


class CustomPipeline(Pipeline):
    # get predict for SHAP (which requires a 2D array as input, even if there's only one sample)
    def predict_shap(self, X):
        probas = super().predict(X)
        if len(X) == 1:
            return [probas]
        else:
            return probas
