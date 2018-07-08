import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X, y = shuffle(X,y, random_state=42)

#X = X[:30]
#y = y[:30]

y[30:] = -1

estimator = KNeighborsClassifier()

estimator.fit(X[:30],y[:30])

print(y)
