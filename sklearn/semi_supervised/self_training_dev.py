import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

est_score_set = []
final_est = []

st_score_set = []
final_st = []

#X, y = load_breast_cancer(return_X_y=True)
X, y = load_iris(return_X_y=True)

for t in tqdm(range(52,62)):
    est_score = []
    st_score = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=t)

    for i in range(10,80):
        lim = i

        y_train_st = y_train.copy()
        y_train_st[lim:] = -1


        est = LogisticRegression()
        est.fit(X_train[:lim], y_train[:lim])
        pred = est.predict(X_test).round()
        est_score.append(accuracy_score(pred, y_test))

        est2 = LogisticRegression()
        st = SelfTraining(est2)
        st.fit(X_train, y_train_st)
        pred = st.predict(X_test).round()
        st_score.append(accuracy_score(pred, y_test))

    est_score_set.append(est_score)
    st_score_set.append(st_score)

est_score_set = np.array(est_score_set).T.tolist()
for l in est_score_set:
    final_est.append(np.mean(l))

st_score_set = np.array(st_score_set).T.tolist()
for l in st_score_set:
    final_st.append(np.mean(l))


plt.figure(1)
plt.plot(final_est, label='Supervised')
plt.plot(final_st, label='Semisupervised')
plt.legend()
plt.show()

