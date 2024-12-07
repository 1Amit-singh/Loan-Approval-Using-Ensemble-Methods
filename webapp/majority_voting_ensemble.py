import numpy as np
from sklearn.base import BaseEstimator

class MajorityVotingEnsemble(BaseEstimator):
    def __init__(self, models):
        """
        Initialize the ensemble with a dictionary of models.
        :param models: dict, keys are model names, values are trained model objects
        """
        self.models = models

    def predict(self, X):
        """
        Predict the final output based on majority voting.
        :param X: np.ndarray, input features
        :return: np.ndarray, final predictions
        """
        predictions = np.array([model.predict(X) for model in self.models.values()])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

    def predict_proba(self, X):
        """
        Predict confidence scores of all models.
        :param X: np.ndarray, input features
        :return: dict, confidence scores mapped to model names
        """
        confidence_scores = {}
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                confidence_scores[name] = model.predict_proba(X)[:, 1]  # Probability for positive class
            else:
                # Assign 0.5 if the model does not support probability prediction
                confidence_scores[name] = np.full(X.shape[0], 0.5)
        return confidence_scores
