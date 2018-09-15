import numpy as np
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.exceptions import NotFittedError

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state

# Author: Oliver Rausch
# License: BSD 3 clause


rng = check_random_state(0)

# load the iris dataset and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    random_state=rng)

limit = 50

y_train_missing_labels = y_train.copy()
y_train_missing_labels[limit:] = -1


def test_classification():
    # Check classification for various parameter settings.
    grid = ParameterGrid({"max_iter": [1, 50, 100],
                          "threshold": [0.0, 0.5, 0.9]})

    for base_estimator in [DummyClassifier(),
                           DecisionTreeClassifier(),
                           KNeighborsClassifier(),
                           SVC(gamma="scale", probability=True)]:
        for params in grid:
            st = SelfTrainingClassifier(base_estimator, **params)
            st.fit(X_train, y_train_missing_labels).predict(X_test)


def test_missing_predict_proba():
    # Check that an error is thrown if predict_proba is not implemented
    base_estimator = SVC(gamma="scale")
    self_training = SelfTrainingClassifier(base_estimator)
    message = "The base_estimator should implement predict_proba!"
    assert_raise_message(ValueError, message, self_training.fit, X_train,
                         y_train)


def test_invalid_params():
    # Test negative iterations
    grid = ParameterGrid({"max_iter": [-1, -100, -10]})
    base_estimator = SVC(gamma="scale", probability=True)
    for params in grid:
        st = SelfTrainingClassifier(base_estimator, **params)
        message = "max_iter must be >= 0"
        assert_raise_message(ValueError, message, st.fit, X_train, y_train)

    grid = ParameterGrid({"threshold": [1.0, -2, 10]})
    base_estimator = SVC(gamma="scale", probability=True)
    for params in grid:
        st = SelfTrainingClassifier(base_estimator, **params)
        message = "threshold must be in [0,1)"
        assert_raise_message(ValueError, message, st.fit, X_train, y_train)


def test_single_iteration():
    # Check classification for single iteration.
    # Fitting a SelfTrainingClassifier with one iteration and 100 unlabeled
    # datapoints should give the same results as fitting a normal classifier
    # with only 50 labeled datapoints.

    clf1 = SelfTrainingClassifier(KNeighborsClassifier(),
                                  max_iter=0).fit(X_train,
                                                  y_train_missing_labels)

    clf2 = KNeighborsClassifier().fit(X_train[:limit], y_train[:limit])

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_notfitted():
    # Test that predicting without training throws an error
    st = SelfTrainingClassifier(KNeighborsClassifier())
    msg = ("This SelfTrainingClassifier instance is not fitted yet. Call "
           "\'fit\' with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg, st.predict, X_train)
    assert_raise_message(NotFittedError, msg, st.predict_proba, X_train)


def test_fitted():
    # Sanity check to see that no errors are thrown
    st = SelfTrainingClassifier(KNeighborsClassifier())
    st.fit(X_train, y_train_missing_labels)
    st.predict(X_train)


def test_y_labeled_iter():
    # Check that the amount of datapoints labeled in iteration 0 is equal to
    # the amount of labeled datapoints we passed.
    for m in range(1, 5):
        st = SelfTrainingClassifier(KNeighborsClassifier(), max_iter=m)
        st.fit(X_train, y_train_missing_labels)
        amount_iter_0 = len(st.y_labeled_iter_[st.y_labeled_iter_ == 0])
        assert(amount_iter_0 == 50)
        # Check that the max of the iterations is less than the total amount of
        # iterations
        assert(np.max(st.y_labeled_iter_) <= m)
