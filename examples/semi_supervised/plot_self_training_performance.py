"""
===================================================
Self-Training: Comparing performance
===================================================
This example demonstrates the performance of the SelfTraining classifier.

The iris dataset is loaded, and a SVC classifier is created. Then, a
SelfTraining classifier is initialised, using the same SVC as its
base estimator.

The dataset contains 150 data points, and the SelfTraining classifier is
trained using all 150 data points, although most are unlabeled. The normal SVC
is trained using only the labeled data points.

The graph shows that the SelfTraining classifier outperforms the normal SVC
when only few labeled data points are available.
"""
# Authors: Oliver Rausch <oliverrausch99@gmail.com>
#          Patrice Becker <patrice@5becker.de>
# License: BSD 3 clause
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised.self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

supervised_score = []
self_training_score = []
x_values = []

clf = SVC(probability=True, C=100, gamma=0.8, kernel='rbf')
self_training_clf = SelfTraining(
    clone(clf, safe=True), max_iter=100, threshold=0.8
)

X, y = load_iris(return_X_y=True)
X, y = shuffle(X, y, random_state=42)
y_true = y.copy()
for t in range(80, 15, -5):
    x_values.append(t)

    lim = t
    y[lim:] = -1

    supervised_score_temp = []
    self_training_score_temp = []

    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        y_test_true = y_true[test_index]

        X_train_filtered = X_train[np.where(y_train != -1)]
        y_train_filtered = y_train[np.where(y_train != -1)]

        clf.fit(X_train_filtered, y_train_filtered)
        y_pred = clf.predict(X_test)
        supervised_score_temp.append(
            f1_score(y_test_true, y_pred, average='macro')
        )

        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        self_training_score_temp.append(
            f1_score(y_test_true, y_pred, average='macro')
        )

    supervised_score.append(np.array(supervised_score_temp).mean())
    self_training_score.append(np.array(self_training_score_temp).mean())

plt.figure(1)
plt.plot(x_values, supervised_score, label='Supervised (SVC)')
plt.plot(x_values, self_training_score, label='SelfTraining')
plt.legend()
plt.ylabel("f1_score (macro average)")
plt.title("Comparison of classifiers on limited labeled data")
plt.xlabel("Amount of Labeled Data")
plt.show()
