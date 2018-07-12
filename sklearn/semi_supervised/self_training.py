#TODO
#rrandomly choose 2p + 2n examples from U to replenish U'
# verify binary (throw error)
# probabilities for p selection
# don't sort

"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, shuffle


class SelfTraining(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, model, ratio_positive=None, n_iter=10, n_iter_size=20, n_iter_insert=5, random_state=None, shuffle_each_iter=True):
        self.model = model
        self.n_iter = n_iter
        self.n_iter_size = n_iter_size
        self.n_iter_insert = n_iter_insert
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

        n_positives = int(round(self.ratio_positive * self.n_iter_insert))
        n_negatives = self.n_iter_insert - n_positives

        for _ in range(self.k):
            insertion_candidates = self._get_random_unlabeled_subset(X,y, self.n_iter_size)
            labeled_X = X[np.where(y != -1)]
            labeled_y = y[np.where(y != -1)]

            self.model.fit(labeled_X, labeled_y)
            pred = np.transpose(self.model.predict_proba(insertion_candidates))
            pred = self.model.predict_proba(insertion_candidates)
            pred = self.model.predict(insertion_candidates)


            # add best positive predictions to the dataset 
            X = np.append(X, insertion_candidates[pred_pos], axis=0)
            y = np.append(y, np.ones(len(best_positives), dtype=int), axis=0)

            # add best negatives predictions to the dataset 
            X = np.append(X, insertion_candidates[pred_neg], axis=0)
            y = np.append(y, np.zeros(len(best_negatives), dtype=int), axis=0)

            if self.shuffle_each_iter:
                X, y = shuffle(X, y, random_state=self.random_state)

        # Return the estimator
        return self.model

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
        pred = self.model.predict(X).round()
        return pred

    def _get_random_unlabeled_subset(self, X, y, size):
        unlabeled_indicies = np.where(y==-1)[0]

        random_choice = check_random_state(self.random_state).choice(unlabeled_indicies, replace=True, size=size)
        return X[random_choice]

    def _detect_ratio_positive(self, y):
        return np.count_nonzero(y)/len(y)
