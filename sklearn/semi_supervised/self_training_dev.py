import numpy as np
import matplotlib.pyplot as plt
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

est_score = []
st_score = []


for i in range(10,60,1):
    print(i)
    lim = i
    X, y = load_breast_cancer(return_X_y=True)
    X, y_testreal = load_breast_cancer(return_X_y=True)
    X, y, y_testreal = shuffle(X,y, y_testreal, random_state=100)
    y[lim:] = -1

    est = KNeighborsClassifier()
    st = SelfTraining(est, u=15, k=20)

    est_score_local = []
    st_score_local = []
    for j in range(1,5):
        #X, y, y_testreal = shuffle(X,y, y_testreal, random_state=j)
        skfolds = StratifiedKFold(n_splits=4, random_state=j)
        for train_index, test_index in skfolds.split(X,y):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            y_test_true = y_testreal[test_index]

            X_train_filtered = X_train[np.where(y_train != -1)]
            y_train_filtered = y_train[np.where(y_train != -1)]

            # get score for supervised
            est.fit(X_train_filtered, y_train_filtered)
            y_pred = est.predict(X_test)
            est_score_local.append(f1_score(y_pred, y_test_true))

            # get score for semi
            st.fit(X_train, y_train)
            y_pred = st.predict(X_test)
            st_score_local.append(f1_score(y_pred, y_test_true))

    est_score.append(np.array(est_score_local).mean())
    st_score.append(np.array(st_score_local).mean())


plt.figure(1)
plt.plot(est_score, label='Supervised')
plt.plot(st_score, label='Semi-supervised')
plt.legend()
plt.show()

