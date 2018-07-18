import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
from self_training import SelfTraining
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised.label_propagation import LabelPropagation, LabelSpreading
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X, y = load_iris(return_X_y=True)
X, y_testreal = load_iris(return_X_y=True)
X, y, y_testreal = shuffle(X,y, y_testreal, random_state=100)

est_score = []
st_score = []
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X =X[:, 2:4]
lim = 25
y[lim:] = -1

C = 100
st = SelfTraining(SVC(kernel='rbf', gamma=0.7, C=C, probability=True), max_iter=1000, threshold=0.7 )
clf = SVC(kernel='rbf', gamma=0.7, C=C)
lbl = LabelSpreading()

st.fit(X,y)
clf.fit(X[:lim], y[:lim])
lbl.fit(X, y)

models = (st, clf, lbl)

titles = ('SVC self training',
          'SVC normal',
          'Labelprop')

fig, sub = plt.subplots(3, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_testreal, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.scatter(X0[:lim], X1[:lim], c=y[:lim], s=20, marker='X', cmap=plt.cm.coolwarm)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
