import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X = X[:20]
y = y[:10]

print(y)

