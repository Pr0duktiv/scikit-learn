"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class SelfTraining(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, model, p=1, n=3, k=30, u=75):
        self.model = model
        self.p = p
        self.n = n
        self.k = k
        self.u = u

    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        U = X[np.where(y == -1)]

        for _ in range(self.k):
            U_small = self._get_random_subbset(X,y)
            L_X = X[np.where(y != -1)]
            L_y = y[np.where(y != -1)]
            self.model.fit(L_X, L_y)
            pred = self.model.predict(U_small)
            pred_pos = np.argpartition(pred, self.p)[-self.p:]
            pred_neg = np.argpartition(pred, self.n)[:self.p]

            X = np.append(X, U_small[pred_pos])
            y = np.append(y, np.ones(len(pred_pos)))

            X = np.append(X, U_small[pred_neg])
            y = np.append(y, np.zeros(len(pred_neg)))

            X, y = shuffle(X, y, random_state=42)



        # Return the estimator
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        X = check_array(X)
        return X[:, 0]**2
