from sklearn.decomposition import sparse_encode
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()
X = iris.data
y = iris.target
V_0 = X[y == 0]
V_1 = X[y == 1]
V_2 = X[y == 2]
numNonZero = 2
tolerance = 1.000000e-05
x1 = np.array([[6.7, 3.5, 2.1, 1.1]])
x2 = np.array([[5.7, 3.3, 3.1, 2.1]])
x3 = np.array([[4.5, 2.5, 2.7, 0.4]])
x4 = np.array([[5.3, 2.4, 4.7, 2.3]])
x5 = np.array([[4.4, 4.0, 5.7, 2.4]])
a = 0.1
y = sparse_encode(x5, V_2, algorithm='omp', n_nonzero_coefs=numNonZero, alpha=tolerance)
temp = np.dot(y, V_2)
cost = np.linalg.norm(x5 - temp) + a * np.count_nonzero(y)
print(cost)
