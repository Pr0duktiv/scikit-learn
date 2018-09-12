import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_X_y, check_array, check_is_fitted
from ..utils.metaestimators import if_delegate_has_method
from ..utils import safe_mask
from ..base import clone

__all__ = ["SelfTrainingClassifier"]


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""

    if not hasattr(estimator, "predict_proba"):
        raise ValueError("The base_estimator should implement predict_proba!")


class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):

    """Self-training classifier

    Parameters
    ----------
    base_estimator : estimator object
        An estimator object implementing `fit` and `predict_proba`.

    threshold : float
        Threshold above which predictions are added to the labeled dataset.
        Should be in [0, 1].

    max_iter : integer
        Maximum number of iterations allowed. Should be greater than or equal
        to 0.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import SelfTrainingClassifier
    >>> from sklearn.svm import SVC
    >>> svc = SVC(probability=True, gamma="auto")
    >>> self_training_model = SelfTrainingClassifier(svc)
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> self_training_model.fit(iris.data, labels)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    SVC(...)

    References
    ----------
    David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling
    supervised methods. In Proceedings of the 33rd annual meeting on
    Association for Computational Linguistics (ACL '95). Association for
    Computational Linguistics, Stroudsburg, PA, USA, 189-196. DOI:
    https://doi.org/10.3115/981658.981684
    """
    def __init__(self,
                 base_estimator,
                 threshold=0.75,
                 max_iter=100):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fits SelfTrainingClassifier to dataset

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            array representing the data
        y : array-like, shape = (n_samples, 1)
            array representing the labels

        Returns
        -------
        self: returns an instance of self.
        """
        X, y = check_X_y(X, y)
        _check_estimator(self.base_estimator)
        self.base_estimator_ = clone(self.base_estimator)

        if not 0 <= self.max_iter:
            raise ValueError("max_iter must be >= 0")

        # Data usable for supervised training
        X_labeled = X[safe_mask(X, np.where(y != -1))][0]
        y_labeled = y[safe_mask(y, np.where(y != -1))][0]

        # Unlabeled data
        X_unlabeled = X[safe_mask(X, np.where(y == -1))][0]
        y_unlabeled = y[safe_mask(y, np.where(y == -1))][0]

        iter = 0
        while len(X_labeled) < len(X) and iter < self.max_iter:
            iter += 1
            self.base_estimator_.fit(X_labeled, y_labeled)

            # Select predictions where confidence is above the threshold
            pred = self.base_estimator_.predict(X_unlabeled)
            max_proba = np.max(self.base_estimator_.predict_proba(X_unlabeled),
                               axis=1)
            confident = np.where(max_proba > self.threshold)[0]

            # Add newly labeled confident predictions to the dataset
            X_labeled = np.append(X_labeled, X_unlabeled[confident], axis=0)
            y_labeled = np.append(y_labeled, pred[confident], axis=0)

            # Remove already labeled data from unlabeled dataset
            X_unlabeled = np.delete(X_unlabeled, confident, axis=0)
            y_unlabeled = np.delete(y_unlabeled, confident, axis=0)

        self.base_estimator_.fit(X_labeled, y_labeled)
        return self

    def predict(self, X):
        """Predict on a dataset.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            array representing the data

        Returns
        -------
        y : array-like, shape = (n_samples, 1)
            array with predicted labels
        """
        check_is_fitted(self, 'base_estimator_')
        X = check_array(X)
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            array representing the data

        Returns
        -------
        y : array-like, shape = (n_samples, n_features)
            array with prediction probabilities
        """
        check_is_fitted(self, 'base_estimator_')
        return self.base_estimator_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self, 'base_estimator_')
        return self.base_estimator_.score(X, y, sample_weight=sample_weight)

    @if_delegate_has_method(delegate='base_estimator')
    def decision_function(self, X):
        check_is_fitted(self, 'base_estimator_')
        return self.base_estimator_.decision_function(X)

    @if_delegate_has_method(delegate='base_estimator')
    def predict_log_proba(self, X):
        check_is_fitted(self, 'base_estimator_')
        return self.base_estimator_.predict_log_proba(X)
