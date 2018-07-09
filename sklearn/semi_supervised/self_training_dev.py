import numpy as np
import matplotlib.pyplot as plt
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

est_score = []
st_score = []

for i in range(20,250):
    lim = i

    X, y = load_breast_cancer(return_X_y=True)
    X, y_testreal = load_breast_cancer(return_X_y=True)
    X, y, y_testreal = shuffle(X,y, y_testreal, random_state=42)
    y[lim:] = -1

    est = KNeighborsClassifier()
    #get_metrics_filter(est)
    est.fit(X[:lim],y[:lim])
    pred = est.predict(X[lim:]).round()
    est_score.append(f1_score(pred, y_testreal[lim:]))
    #print('Supervised Accucary: %f' % accuracy_score(pred, y_testreal[lim:]))

    st = SelfTraining(est, u=15, k=20)
    st.fit(X, y)
    pred = st.predict(X[lim:]).round()
    st_score.append(f1_score(pred, y_testreal[lim:]))
    #print('Self Training Accucary: %f' % accuracy_score(pred, y_testreal[lim:]))

plt.figure(1)
plt.plot(est_score, label='Supervised')
plt.plot(st_score, label='Semisupervised')
plt.legend()
plt.show()

