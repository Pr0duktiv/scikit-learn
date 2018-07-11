import numpy as np
import matplotlib.pyplot as plt
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

est_score = []
st_score = []

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


for i in range(10,50):
    lim = i

    y_train_st = y_train.copy()
    y_train_st[lim:] = -1

    est = KNeighborsClassifier()
    est.fit(X_train[:lim],y_train[:lim])
    pred = est.predict(X_test).round()
    est_score.append(f1_score(pred, y_test))
    #print('Supervised Accucary: %f' % accuracy_score(pred, y_testreal[lim:]))

    st = SelfTraining(est, u=15, k=20)
    st.fit(X_train, y_train_st)
    pred = st.predict(X_test).round()
    st_score.append(f1_score(pred, y_test))
    #print('Self Training Accucary: %f' % accuracy_score(pred, y_testreal[lim:]))

plt.figure(1)
plt.plot(est_score, label='Supervised')
plt.plot(st_score, label='Semisupervised')
plt.legend()
plt.show()

