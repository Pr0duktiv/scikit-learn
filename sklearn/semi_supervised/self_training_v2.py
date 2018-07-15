import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SelfTraining(BaseEstimator):
    def __init__(self, model, treshold=0.9, max_iter=100):
        self.model = model
        self.treshold = treshold
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # Data usable for supervised training
        X_labeled = X[np.where(y != -1)]
        y_labeled = y[np.where(y != -1)]

        # Unlabeled data
        X_unlabeled = X[np.where(y == -1)]
        y_unlabeled = y[np.where(y == -1)]

        iter = 0
        while (len(X_labeled) < len(X) and iter < self.max_iter):
            iter += 1
            self.model.fit(X_labeled, y_labeled)
            pred_all = self.predict(X_unlabeled)
            max_proba = np.max(self.predict_proba(X_unlabeled), axis=1)
            pred_confident = np.where(max_proba > self.treshold)[0]

            # Add newly labeled data to labeled dataset
            X_labeled = np.append(X_labeled, X_unlabeled[pred_confident], axis=0)
            y_labeled = np.append(y_labeled, pred_all[pred_confident].round(), axis=0)

            # Remove already labeled data from unlabeled dataset
            X_unlabeled = np.delete(X_unlabeled, pred_confident, axis=0)
            y_unlabeled = np.delete(y_unlabeled, pred_confident, axis=0)

        self.model.fit(X_labeled, y_labeled)
        return self.model

    def predict(self, X):
        X = check_array(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise TypeError('"predict_proba" not implemented for model %s' % type(self.model))
