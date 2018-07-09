import numpy as np
import matplotlib as plt
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def get_metrics_filter(estimator):
    X, y = load_iris(return_X_y=True)
    X, y_testreal = load_iris(return_X_y=True)
    X, y, y_testreal = shuffle(X,y, y_testreal, random_state=42)
    y[:30] = -1
    skfolds = StratifiedKFold(n_splits=4, random_state=42)

    for train_index, test_index in skfolds.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        y_test_true = y_testreal[test_index]


        X_train_filtered = X_train[np.where(y_train != -1)]
        y_train_filtered = y_train[np.where(y_train != -1)]
       # print("Xtrain:" , X_train)
       # print("y_train:", y_train)

       # print("Xtrain_filt:" , X_train_filtered)
       # print("y_train_filt:", y_train_filtered)

        estimator.fit(X_train_filtered, y_train_filtered)
        y_pred = estimator.predict(X_test)
        print(accuracy_score(y_pred, y_test_true))

estimator = KNeighborsClassifier()

get_metrics(estimator)



