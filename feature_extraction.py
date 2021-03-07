from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np


iris = datasets.load_iris()
X = iris.data
y = iris.target
n_samples, n_features = X.shape
# calculate mean of all feature vectors
mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
# normalisation
norm_X = X - mean
# calculate covariance matrix
cov_matrix = np.dot(np.transpose(norm_X), norm_X)
# calculate eigenvalues and eigenvectors
eig_val, eig_vec = np.linalg.eig(cov_matrix)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
# sort eig_vec based on eig_val from highest to lowest
eig_pairs.sort(reverse=True)
# select the top k eigenvectors
k = 2
feature = np.array([ele[1] for ele in eig_pairs[:k]])
# get new data
sample_5 = np.array([7.1, 3.3, 6.0, 2.1])
norm_5 = sample_5 - mean
data = np.dot(norm_5, np.transpose(feature))
print(data)
