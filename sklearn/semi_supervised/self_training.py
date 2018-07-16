import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import safe_mask

def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("The base estimator should implement predict_proba!")

class SelfTraining(BaseEstimator):
    def __init__(self, estimator, threshold=0.7, max_iter=500):
        self.estimator = estimator
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        _check_estimator(self.estimator)

        # Data usable for supervised training
        X_labeled = X[np.where(y != -1)]
        y_labeled = y[np.where(y != -1)]

        # Unlabeled data
        X_unlabeled = X[np.where(y == -1)]
        y_unlabeled = y[np.where(y == -1)]

        iter = 0
        while (len(X_labeled) < len(X) and iter < self.max_iter):
            iter += 1
            self.estimator.fit(X_labeled, y_labeled)
            pred_all = self.predict(X_unlabeled)
            max_proba = np.max(self.predict_proba(X_unlabeled), axis=1)
            pred_confident = np.where(max_proba > self.threshold)[0]

            # Add newly labeled data to labeled dataset
            X_labeled = np.append(X_labeled, X_unlabeled[pred_confident], axis=0)
            y_labeled = np.append(y_labeled, pred_all[pred_confident].round(), axis=0)

            # Remove already labeled data from unlabeled dataset
            X_unlabeled = np.delete(X_unlabeled, pred_confident, axis=0)
            y_unlabeled = np.delete(y_unlabeled, pred_confident, axis=0)

        self.estimator.fit(X_labeled, y_labeled)
        return self.estimator

    def predict(self, X):
        check_is_fitted(self, 'estimator')
        X = check_array(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        _check_estimator(self.estimator)
        check_is_fitted(self, 'estimator')
        return self.estimator.predict_proba(X)
