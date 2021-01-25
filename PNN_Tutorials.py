# Negative Feedback Network
import numpy as np
# W = np.array([[1, 1, 0], [1, 1, 1]])
# x = np.array([1, 1, 0])
# n = 5
# y = np.array([[0, 0, 0], [0, 0, 0]])
# i = 1
# while i <= n:
#    e = x.T - np.dot(W.T, y)
#    y = y + 0.5 * np.dot(W, e)
#    i = i + 1
# print(y)


# Tutorial 2.1 dichotomy
# def linear_discriminant(w, x, w0):
#    g = np.dot(w, x) + w0
#    return g


# W = np.array([2, 1])
# W0 = -5
# X = np.array([3, 3])
# g_x = linear_discriminant(W, X, W0)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.2, 2.5 dichotomy in augmented feature space
# def aug_linear_discriminant(a, x):
#    y = np.insert(x, 0, 1)
#    g = np.dot(a, y)
#    return g


# a_t = np.array([-3, 1, 2, 2, 2, 4])
# X = np.array([1, 1, 1, 1, 1])
# g_x = aug_linear_discriminant(a_t, X.T)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.3 3D quadratic discriminant
# def quadratic_discriminant_3d(x):
#    g = pow(x[0], 2) - pow(x[2], 2) + 2 * x[1] * x[2] + 4 * x[0] * x[1] + 3 * x[0] - 2 * x[1] + 2
#    return g


# X = np.array([-1, 0, 0])
# g_x = quadratic_discriminant_3d(X)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.4 2D quadratic discriminant
# def quadratic_discriminant_2d(a, b, c, x):
#    g = np.dot(np.dot(x, a), x.T) + np.dot(x, b.T) + c
#    return g


# A = np.array([[2, 1], [1, 4]])
# B = np.array([1, 2])
# C = -3
# X = np.array([1, 1])
# g_x = quadratic_discriminant_2d(A, B, C, X)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.6, 2.9, 2.10 sequential perceptron
# def sequential_perceptron(a, lr, yk, labels):
#    while True:
#        pred_label = []
#        for i in range(len(yk)):
#            g = np.dot(a, yk[i].T)
#            if g > 0:
#                pre_label = 1
#            else:
#                pre_label = -1
#            pred_label.append(pre_label)
#            if pre_label != labels[i]:
#                a = a + lr * labels[i] * yk[i]
#        if pred_label == labels:
#            break
#    return a


# if __name__ == '__main__':
#    a_t = np.array([1, 0, 0])
#    rate = 1
#    data_set = np.array([[1, 0, 2],
#                         [1, 1, 2],
#                         [1, 2, 1],
#                         [1, -3, 1],
#                         [1, -2, -1],
#                         [1, -3, -2]])
#    label = [1, 1, 1, -1, -1, -1]
#    a_final = sequential_perceptron(a_t, rate, data_set, label)
#    print("The value of a after learning is: {}.".format(a_final))


# Tutorial 2.12, 2.13 pseudoinverse
# Y = np.array([[1, 0, 2],
#              [1, 1, 2],
#              [1, 2, 1],
#             [1, -3, 1],
#             [1, -2, -1],
#              [1, -3, -2]])  # Y should be in sample normalised format (类别2的标签w2 * -1)
# b = np.array([1, 1, 1, 2, 2, 2])
# a = np.dot(np.linalg.pinv(Y), b.T)
# print("a = {}".format(a))


# Tutorial 2.15 KNN
# from sklearn import datasets  # import data sets module from sklearn library
# from sklearn.neighbors import KNeighborsClassifier  # import KNN classifier model from sklearn's neighbors module
# knn = KNeighborsClassifier(n_neighbors=3)  # Define KNN classier model (n_neighbors represents the value of k)
# y = np.array([1, 2, 2, 3, 3])  # specify prediction target
# X = np.array([[0.15, 0.35],
#              [0.15, 0.28],
#              [0.12, 0.2],
#              [0.1, 0.32],
#              [0.06, 0.25]])  # choose features, y = f(X)
# knn.fit(X, y)  # fit the model (the heart of modelling)
# print(knn.predict([[0.1, 0.25]]))  # using this model to predict
