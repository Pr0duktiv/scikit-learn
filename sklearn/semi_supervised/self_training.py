#TODO
#rrandomly choose 2p + 2n examples from U to replenish U'
# verify binary (throw error)
# probabilities for p selection

"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state


class SelfTraining(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, model, ratio_positive=None, k=30, u=7, random_state=None, shuffle_each_iter=True):
        self.model = model
        self.k = k
        self.u = u
        self.random_state = random_state
        self.shuffle_each_iter = shuffle_each_iter
        self.ratio_positive = ratio_positive

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

        if self.ratio_positive == None:
            self.ratio_positive = self._detect_ratio_positive(y)

        p = int(round(ratio_positive * len(y)))
        n = len(y) - p

        U = X[np.where(y == -1)]

        for _ in range(self.k):
            U_small = self._get_random_subset(X,y, self.u)
            L_X = X[np.where(y != -1)]
            L_y = y[np.where(y != -1)]
            self.model.fit(L_X, L_y)
            pred = self.model.predict(U_small)
            pred_pos = np.argpartition(pred, p)[-p:]
            pred_neg = np.argpartition(pred, n)[:n]

            X = np.append(X, U_small[pred_pos])
            y = np.append(y, np.ones(len(pred_pos)))

            X = np.append(X, U_small[pred_neg])
            y = np.append(y, np.zeros(len(pred_neg)))

            if shuffle_each_iter:
                X, y = shuffle(X, y, random_state=self.random_state)

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

    def _get_random_subset(self, X, y, size):
        unlabeled_indicies = np.where(y==-1)[0]

        random_choice = check_random_state(self.random_state).choice(unlabeled_indicies, replace=True, size=size)

        return X[random_choice], y[random_choice]

    def _detect_ratio_positive(self, y):
        return np.count_nonzero(y)/len(y)



