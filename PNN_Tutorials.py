import numpy as np
import torch
from torch.nn import ReLU, LeakyReLU, Tanh, AvgPool2d, MaxPool2d
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Tutorial 2.1 linear discriminant dichotomizer
# def linear_discriminant(w, x, w0):
#    g = np.dot(w, x) + w0
#    return g


# W = np.array([-1, -2])
# W0 = 3
# X = np.array([2.5, 1.5])
# g_x = linear_discriminant(W, X, W0)
# print(g_x)  # 如果不是dichotomizer, 不同的a_t对应不同的g_x, 取最大的g_x所对应的类别
# if g_x > 0:
#    print("The class of this vector is 1.")
# else:
#    print("The class of this vector is 2.")


# Tutorial 2.2, 2.5 augmented linear discriminant dichotomizer
# def aug_linear_discriminant(a, x):
#    y = np.insert(x, 0, 1)
#    g = np.dot(a, y)
#    return g


# a_t = np.array([-2, 1, 2, 1, 2])
# X = np.array([0, -1, 0, 1])
# g_x = aug_linear_discriminant(a_t, X.T)
# print(g_x)  # 如果不是dichotomizer, 不同的a_t对应不同的g_x, 取最大的g_x所对应的类别
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.3 3-dimensional quadratic discriminant dichotomizer
# def quadratic_discriminant_3d(x):
#    g = x[0] ** 2 - x[2] ** 2 + 2 * x[1] * x[2] + 4 * x[0] * x[1] + 3 * x[0] - 2 * x[1] + 2
#    return g


# X = np.array([1, 1, 1])
# g_x = quadratic_discriminant_3d(X)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.4 2-dimensional quadratic discriminant dichotomizer
# def quadratic_discriminant_2d(a, b, c, x):
#    g = np.dot(np.dot(x, a), x.T) + np.dot(x, b.T) + c
#    return g


# A = np.array([[2, 1], [1, 4]])
# B = np.array([1, 2])
# C = -3
# X = np.array([1, 1])
# g_x = quadratic_discriminant_2d(A, B, C, X)
# if g_x > 0:
#    print("The class of this vector is 1.")
# else:
#    print("The class of this vector is 2.")


# Tutorial 2.6 batch perceptron
# def batch_perceptron_with_sample_normalisation(a, lr, y):
#    while True:
#        mis_class = []
#        y_mis = []
#        temp = 0
#        for i in range(len(y)):
#            g = np.dot(a, y[i].T)
#            if g > 0:
#                mis_class.append('no')
#            else:
#                mis_class.append('yes')
#                y_mis.append(y[i])

        # when g(x) > 0 for all y, the algorithm converges
#        if mis_class.count('no') != len(mis_class):
#            for j in range(len(y_mis)):
#                temp += lr * y_mis[j]
#            a += np.array(temp)
#        else:
#            break
#    return a


# def batch_perceptron(a, lr, y, labels):
#    while True:
#        pred_label = []
#        y_mis = []
#        temp = 0
#        for i in range(len(y)):
#            g = np.dot(a, y[i].T)
#            if g > 0:
#                pre_label = 1
#            else:
#                pre_label = 2
#            pred_label.append(pre_label)
#            if pre_label != labels[i]:
#                y_mis.append(y[i])
#        if pred_label == labels:
#            break
#        else:
#            for j in range(len(y_mis)):
#                temp += lr * y_mis[j]
#            a += np.array(temp)
#    return a

# a = np.array([-25, 6, 3])
# lr = 1

# 没有sample normalisation, 将x的第一位插入1得到y;
# 如果有, 则第二类实例的y的第一位是-1并且x乘以-1, a的更新条件也发生变化

# y = np.array([[1, 1, 5],
#              [1, 2, 5],
#              [-1, -4, -1],
#              [-1, -5, -1]])
# label = [1, 1, 2, 2]
# a_final = batch_perceptron_with_sample_normalisation(a, lr, y)
# a_final = batch_perceptron(a, lr, y, label)
# print("The value of a after learning is:", a_final)


# Tutorial 2.7, 2.9, 2.10 sequential perceptron
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


# def sequential_perceptron_with_sample_normalisation(a, lr, yk):
#    while True:
#        count = 0
#        for i in range(len(yk)):
#            g = np.dot(a, yk[i].T)
#            if g <= 0:
#                a = a + lr * yk[i]
#            else:
#                count += 1
#        if len(yk) == count:
#            break
#    return a


# if __name__ == '__main__':
#    a_t = np.array([-1.5, 5, -1])
#    rate = 1

#    没有sample normalisation, y的第一位都是1;
#    如果有, 则第二类实例的y的第一位是-1且x乘以-1, 并且a的更新条件和更新等式都发生变化

#    data_set = np.array([[1, 0, 0],
#                         [1, 1, 0],
#                         [1, 2, 1],
#                         [1, 0, 1],
#                         [1, 1, 2]])
#    label = [1, 1, 1, -1, -1]  # true class (target output)
#    a_final = sequential_perceptron(a_t, rate, data_set, label)
#    a_final = sequential_perceptron_with_sample_normalisation(a_t, rate, data_set)
#    print("The value of a after learning is:", a_final)


# Tutorial 2.11 sequential multiclass perceptron
# def sequential_multiclass_perceptron(a, y, lr, labels):
#    n = 0
#    while True:
#        for i in range(len(y)):
#            g1 = np.dot(a[0], y[i])
#            g2 = np.dot(a[1], y[i])
#            g3 = np.dot(a[2], y[i])
#            if g1 == g2 == g3:
#                a[2] -= lr * y[i]
#                a[labels[i]-1] += lr * y[i]
#            if g1 == max(g1, g2, g3):
#                if labels[i] != 1:
#                    a[labels[i]-1] += lr * y[i]
#                    a[0] -= lr * y[i]
#            elif g2 == max(g1, g2, g3):
#                if labels[i] != 2:
#                    a[labels[i]-1] += lr * y[i]
#                    a[1] -= lr * y[i]
#            elif g3 == max(g1, g2, g3):
#                if labels[i] != 3:
#                    a[labels[i]-1] += lr * y[i]
#                    a[2] -= lr * y[i]
#            n += 1
#            print("n = {0}, a1 = {1}, a2 = {2}, a3 = {3}".format(n, a[0], a[1], a[2]))


# a_t = np.array([[1, 0.5, 0.5, -0.75], [-1, 2, 2, 1], [2, -1, -1, 1]])
# 记得将x的第一位插入1得到y
# y_t = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [1, 0.5, 0.5, 0.25], [1, 1, 1, 1], [1, 0, 0, 0]])
# rate = 1
# label = [1, 1, 2, 2, 3]
# sequential_multiclass_perceptron(a_t, y_t, rate, label)  # press Control-C to stop


# Tutorial 2.12, 2.13 pseudoinverse with sample normalisation
# Y = np.array([[1, 0, 2],
#              [1, 1, 2],
#              [1, 2, 1],
#              [-1, 3, -1],
#              [-1, 2, 1],
#              [-1, 3, 2]])
# b = np.array([2, 2, 2, 1, 1, 1])
# a = np.dot(np.linalg.pinv(Y), b.T)
# print("a =", a)

# Tutorial 2.14 sequential widrow hoff
# 参考 Sequential_Widrow_Hoff.py

# Tutorial 2.15 KNN k-nearest-neighbor
# knn = KNeighborsClassifier(n_neighbors=5)
# y = np.array([1, 1, 1, 2, 3])  # prediction target
# X = np.array([[-2, 6],
#              [-1, -4],
#              [3, -1],
#              [-3, -2],
#              [-4, -5]])
# knn.fit(X, y)
# print(knn.predict([[-2, 0]]))


# Tutorial 3.2 Heaviside function
# 注意题目是否提到augmented notation, 如果提到, w = [w0, w1, ...], w0 = -theta, 且x的第一位插入1
# def heaviside(w, x):
#    threshold = -2
#    y = np.dot(w, x.T) - threshold
#    y = np.dot(w, x.T)  # augmented
#    if y >= 0:
#        print("The output of neuron is: 1")
#    else:
#        print("The output of neuron is: 0")


# theta = -2
# W = np.array([-theta, 0.5, 1])
# x = np.array([1, 0, -1])
# heaviside(W, x)


# Tutorial 3.3, 3.5, 3.6 Sequential Delta
# augmented notation w = [w0, w1, ..., wn], w0=-theta
# def sequential_delta(w, lr, t, x):
#    n = 0
#    while True:
#        y = []
#        for i in range(len(x)):
#            h = np.dot(w, np.insert(x[i].T, 0, 1))
#            if h > 0:
#                y.append(1)
#            else:
#                y.append(0)
#            w = w + lr * (t[i] - y[i]) * np.insert(x[i], 0, 1)
#            n += 1
#            print("n = {0}, y = {1}, w = {2}".format(n, y[i], w))
#        if y == t:
#            break


# Tutorial 3.4 batch Delta
# def batch_delta(w, lr, t, x):
#    epoch = 0
#    while True:
#        y = []
#        Temp = []
#        for i in range(len(x)):
#            h = np.dot(w, np.insert(x[i].T, 0, 1))
#            if h > 0:
#                y.append(1)
#            else:
#                y.append(0)
#            temp = lr * (t[i] - y[i]) * np.insert(x[i], 0, 1)
#            Temp.append(temp)
#        w += sum(Temp)
#        epoch += 1
#        print("epoch = {0}, w = {1}".format(epoch, w))
#        if y == t:
#            break


# theta = 1.5
# w1 = 5
# w2 = -1
# rate = 1
# W = np.array([-theta, w1, w2])
# T = [1, 1, 1, 0, 0]  # class y (target output)
# X = np.array([[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]])  # 直接输入x的值
# sequential_delta(W, rate, T, X)
# batch_delta(W, rate, T, X)


# Tutorial 3.7, 3.8 negative feedback network
# def negative_feedback(x, w, alpha, n):
#    y = np.array([0, 0])  # assuming activations are initialised to be all 0
#    for i in range(0, n):
#        e = x.T - np.dot(w.T, y.T)
#        y = y + alpha * np.dot(w, e)
#        print("i = {0}, the activation of the output neurons: {1}".format(i+1, y))


# Tutorial 3.9 negative feedback network more stable method
# def negative_feedback_stable(x, w1, w2, n, e1, e2):
#    y = np.array([0, 0])  # assuming activations are initialised to be all 0
#    for i in range(0, n):
#        a = []
#        b = []
#        temp = np.dot(w1.T, y.T)
#        for j in range(len(temp)):
#            a.append(max(temp[j], e2))
#        e = x.T / np.array(a)
#        for k in range(len(y)):
#            b.append(max(y[k], e1))
#        y = np.array(b) * np.dot(w2, e)
#        print("i = {0}, the activation of the output neurons: {1}".format(i + 1, y))


# X = np.array([1, 1, 0])
# Alpha = 0.25
# W = np.array([[1, 1, 0],
#              [1, 1, 1]])
# w = np.array([[1/2, 1/2, 0],
#              [1/3, 1/3, 1/3]])
# count = 4  # iterations
# e_1 = 0.01
# e_2 = 0.01
# negative_feedback_stable(X, W, w, count, e_1, e_2)
# negative_feedback(X, W, Alpha, count)


# Tutorial 4.1 total connection weights
# def total_connection_weights(num_in, num_out, hidden, num_hid, bias):
#    num_weights = 0
#    if bias:
#        for i in range(num_hid):
#            if i == 0:
#                num_weights += (num_in + 1) * hidden[i]
#            else:
#                num_weights += ((hidden[i - 1]) + 1) * hidden[i]
#        num_weights += (hidden[num_hid - 1] + 1) * num_out
#    else:
#        for i in range(num_hid):
#            if i == 0:
#                num_weights += num_in * hidden[i]
#            else:
#                num_weights += hidden[i - 1] * hidden[i]
#        num_weights += hidden[num_hid - 1] * num_out
#    return num_weights


# n_in = 2
# n_out = 3
# hid = [4, 5]  # the number of hidden units
# num_hidden = 2  # the number of hidden layers
# Bias = True  # with biases, True; otherwise, False
# print("Total number of weights:", total_connection_weights(n_in, n_out, hid, num_hidden, Bias))


# Tutorial 4.4 neural network classify patterns
# def represent_patterns(w1, w2, w3, w4, x):
#    temp_y = np.dot(w1, x) + w2
#    y = 2 / (1 + 1 / pow(math.e, 2 * temp_y)) - 1  # symmetric sigmoid
#    temp_z = np.dot(w3, y) + w4
#    z = 1 / (1 + 1 / (pow(math.e, temp_z)))  # logarithmic sigmoid
#    print("y = {0}, z = {1}".format(y, z))
#    print("This pattern is represented by", [round(i) for i in z])


# W1 = np.array([[-0.7057, 1.9061, 2.6605, -1.1359],
#               [0.4900, 1.9324, -0.4269, -5.1570],
#               [0.9438, -5.4160, -0.3431, -0.2931]])  # W_ji
# W2 = np.array([4.8432, 0.3973, 2.1761])  # W_j0
# W3 = np.array([[-1.1444, 0.3115, -9.9812], [0.0106, 11.5477, 2.6479]])  # W_kj
# W4 = np.array([2.5230, 2.6463])  # W_k0
# X = np.array([0, 1, 0, 1])  # pattern
# represent_patterns(W1, W2, W3, W4, X)


# Tutorial 4.5 neural network
# Q.a
# w = np.array([[0.5, 0], [0.3, -0.7]])  # w11, w12; w21, w22
# w0 = np.array([0.2, 0])  # w10, w20
# x = np.array([0.1, 0.9])
# temp_y = np.dot(w, x.T) + w0
# y = 2 / (1 + 1 / pow(math.e, 2 * temp_y)) - 1  # symmetric sigmoid
# temp_z = 0.8 * y[0] + 1.6 * y[1] - 0.4
# z = 2 / (1 + 1 / pow(math.e, 2 * temp_z)) - 1  # symmetric sigmoid
# print("y = {0}, z = {1}".format(y, z))

# Q.c
# lr = 0.25
# t = 0.5
# m10 = -0.4
# temp_m = -lr * (z - t) * (4 / pow(math.e, 2 * temp_z)) / pow(1 + 1 / pow(math.e, 2 * temp_z), 2)
# m10 += temp_m
# w21 = 0.3
# w22 = -0.7
# m12 = 1.6
# a = w21 * x[0] + w22 * x[1]
# temp_w = -lr * (z - t) * ((4 / pow(math.e, 2 * temp_z)) / pow(1 + 1 / pow(math.e, 2 * temp_z), 2)) \
#         * m12 * ((4 / pow(math.e, 2 * a)) / pow(1 + 1 / pow(math.e, 2 * a), 2)) * x[0]
# w21 += temp_w
# print("m10 = {0}, w21 = {1}".format(m10, w21))


# Tutorial 4.6 radial basis function RBF
# Q.b
# c1, c2 = np.array([0, 0]), np.array([1, 1])  # centres
# c = abs(c1 - c2)
# n_H = 2  # the number of hidden units
# x = np.array([0, 0])  # two inputs
# sd = np.linalg.norm(c) / math.sqrt(2 * n_H)
# d1 = abs(x - c1)
# y1 = pow(math.e, -pow(np.linalg.norm(d1), 2) / (2 * pow(sd, 2)))  # Gaussian function
# d2 = abs(x - c2)
# y2 = pow(math.e, -pow(np.linalg.norm(d2), 2) / (2 * pow(sd, 2)))  # Gaussian function
# print("y1 = {0}, y2 = {1}".format(y1, y2))

# Q.c least squares method
# 代入不同的x, 分别计算出y1和y2; fai_x的第一列是y1, 第二列是y2, 第三列是1 (bias)
# fai_x = np.array([[1.0, 0.1353, 1], [0.3679, 0.3679, 1], [0.3679, 0.3679, 1], [0.1353, 1.0, 1]])
# t = np.array([0, 1, 1, 0])
# temp1 = np.dot(fai_x.T, t)
# temp2 = np.dot(fai_x.T, fai_x)
# W = np.dot(np.linalg.inv(temp2), temp1)
# print("The output weights using the least squares method is:", W)

# Q.d determine classes
# W = [-2.50312891, -2.50312891, 2.84180225]  # 用Q.c中计算出的权重值W
# z = []
# Class = []
# for i in range(len(fai_x)):
#    z.append(W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2])
# print("The outputs at the output units:\n", z)
# for i in range(len(z)):
#    if z[i] > 0.5:
#        Class.append('1')
#    else:
#        Class.append('0')
# print("The classes for the input patterns are:", ' '.join(Class))


# Tutorial 4.7
# Q.a
# The answer is located at Jupyter Notebook

# Q.c, Q.d, Q.e RBF
# c1, c2, c3 = 0.2, 0.6, 0.9
# sd = 0.1
# x = [0.05, 0.2, 0.25, 0.3, 0.4, 0.43, 0.48, 0.6, 0.7, 0.8, 0.9, 0.95]  # single input
# t = np.array([0.0863, 0.2662, 0.2362, 0.1687, 0.1260, 0.1756, 0.3290, 0.6694, 0.4573, 0.3320, 0.4063, 0.3535])
# c1 = (x[0] + x[1] + x[2]) / 3
# c2 = (x[3] + x[4]) / 2
# c3 = (x[5] + x[6] + x[7] + x[8]) / 4
# c4 = (x[9] + x[10] + x[11]) / 3
# avg = (abs(c1-c2) + abs(c1-c3) + abs(c1-c4) + abs(c2-c3) + abs(c2-c4) + abs(c3-c4)) / 6
# sd = 2 * avg
# fai_x = []
# for i in range(len(x)):
#    y1 = pow(math.e, -pow(x[i] - c1, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y1)
#    y2 = pow(math.e, -pow(x[i] - c2, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y2)
#    y3 = pow(math.e, -pow(x[i] - c3, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y3)
#    y4 = pow(math.e, -pow(x[i] - c4, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y4)
#    fai_x.append(1)
#    print("p = {}, y1 = {}, y2 = {}, y3 = {}".format(i + 1, y1, y2, y3))
#    print("p = {}, y1 = {}, y2 = {}, y3 = {}, y4 = {}".format(i + 1, y1, y2, y3, y4))
# fai_x = np.array(fai_x)
# fai_x = fai_x.reshape((len(x), 4))  # 列数是y1, y2...的个数+1
# temp1 = np.dot(fai_x.T, t)
# temp2 = np.dot(fai_x.T, fai_x)
# W = np.dot(np.linalg.inv(temp2), temp1)
# print("The output weights are:", W)
# J = []
# for i in range(len(x)):
#    z = W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2] * fai_x[i][2] + W[3] * fai_x[i][3] + W[4]
#    z = W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2] * fai_x[i][2] + W[3]
#    J.append(pow((z - t[i]), 2))
#    J[i] = round(J[i], ndigits=4)
#    print(J[i])
# print("The sum of squared errors is:", sum(J))


# Tutorial 5.4 activation functions
# ReLU
# net = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# net_tensor = torch.from_numpy(net)
# relu = ReLU()
# print(relu(net_tensor))

# LReLU when a = 0.1
# a = 0.1
# lrelu = LeakyReLU(a)
# print(lrelu(net_tensor))

# tanh
# tanh = Tanh()
# print(tanh(net_tensor))

# Heaviside function with 0.1 threshold and define H(0) as 0.5
# for i in range(len(net)):
#    for j in range(len(net[i])):
#        if net[i][j] - 0.1 > 0:
#            net[i][j] = 1
#        elif net[i][j] - 0.1 < 0:
#            net[i][j] = 0
#        elif net[i][j] - 0.1 == 0:
#            net[i][j] = 0.5
# print(net)


# Tutorial 5.5 batch normalisation
# def batch_normalisation(x1, x2, x3, x4, beta, r, e):
#    bn1, bn2, bn3, bn4 = [], [], [], []
#    for i in range(len(x1)):
#        for j in range(len(x1[i])):
#            avg = np.mean([x1[i][j], x2[i][j], x3[i][j], x4[i][j]])
#            var = np.var([(x1[i][j] - avg), (x2[i][j] - avg), (x3[i][j] - avg), (x4[i][j] - avg)])
#            temp = beta + r * (x1[i][j] - avg) / math.sqrt(var + e)
#            bn1.append(temp)
#            temp = beta + r * (x2[i][j] - avg) / math.sqrt(var + e)
#            bn2.append(temp)
#            temp = beta + r * (x3[i][j] - avg) / math.sqrt(var + e)
#            bn3.append(temp)
#            temp = beta + r * (x4[i][j] - avg) / math.sqrt(var + e)
#            bn4.append(temp)
#    bn1 = np.array(bn1).reshape(3, 3)
#    bn2 = np.array(bn2).reshape(3, 3)
#    bn3 = np.array(bn3).reshape(3, 3)
#    bn4 = np.array(bn4).reshape(3, 3)
#    print("The Batch Normalisation of X1 is: \n {}".format(bn1))
#    print("The Batch Normalisation of X2 is: \n {}".format(bn2))
#    print("The Batch Normalisation of X3 is: \n {}".format(bn3))
#    print("The Batch Normalisation of X4 is: \n {}".format(bn4))


# b, r, e = 0, 1, 0.1
# x_1 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# x_2 = np.array([[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]])
# x_3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
# x_4 = np.array([[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]])
# batch_normalisation(x_1, x_2, x_3, x_4, b, r, e)


# Tutorial 5.6 feature maps, convolution layer, padding, stride
# a. padding = 0, stride = 1
# X1 = np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]])
# X2 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# H1 = np.array([[1, -0.1], [1, -0.1]])
# H2 = np.array([[0.5, 0.5], [-0.5, -0.5]])
# out1 = []
# out2 = []
# for i in range(len(X1) - 1):
#    for j in range(len(X1)):
#        if j + 2 < len(X1) + 1:
#            out1.append(sum(sum(X1[i: i + 2, j: j + 2] * H1)))
#            out2.append(sum(sum(X2[i: i + 2, j: j + 2] * H2)))
#        else:
#            break
# out1 = np.array(out1).reshape(2, 2)  # 输出结构根据X和H决定
# out2 = np.array(out2).reshape(2, 2)
# out = out1 + out2
# print("The result of padding = 0 and stride = 1 is:\n", out)

# b. padding = 1, stride = 1
# X1_pad = np.pad(X1, 1)
# X2_pad = np.pad(X2, 1)
# out1_pad = []
# out2_pad = []
# for i in range(len(X1_pad) - 1):
#    for j in range(len(X1_pad)):
#        if j + 2 < len(X1_pad) + 1:
#            out1_pad.append(sum(sum(X1_pad[i: i + 2, j: j + 2] * H1)))
#            out2_pad.append(sum(sum(X2_pad[i: i + 2, j: j + 2] * H2)))
#        else:
#            break
# out1_pad = np.array(out1_pad).reshape(4, 4)
# out2_pad = np.array(out2_pad).reshape(4, 4)
# out_pad = out1_pad + out2_pad
# print("The result of padding = 1 and stride = 1 is:\n", out_pad)

# c. padding = 1, stride = 2 (2 step size along the width and height direction)
# using answer to b. part
# out_stride = [out_pad[0, 0], out_pad[0, 2], out_pad[2, 0], out_pad[2, 2]]
# out_stride = np.array(out_stride).reshape(2, 2)
# print("According to the answer to the b part, the result of padding = 1 and stride = 2 is:\n", out_stride)

# d. padding = 0, stride = 1, dilation = 2
# H1_d = np.array([[1, 0, -0.1], [0, 0, 0], [1, 0, -0.1]])
# H2_d = np.array([[0.5, 0, 0.5], [0, 0, 0], [-0.5, 0, -0.5]])
# out_d = X1 * H1_d + X2 * H2_d
# out_d = sum(sum(out_d))
# print("The result of padding = 0, stride = 1 and dilation = 2 is:\n", out_d)


# Tutorial 5.7 feature maps, convolution layer
# x1 = np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]])
# x2 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# x3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
# h = [1, -1, 0.5]
# y = []
# for i in range(len(x1)):
#    for j in range(len(x1)):
#        y.append(x1[i][j] * h[0] + x2[i][j] * h[1] + x3[i][j] * h[2])
# y = np.array(y).reshape(3, 3)
# print(y)


# Tutorial 5.8 pooling layer
# a. average pooling with a pooling region of 2-by-2 (kernel size) and stride = 2
# x = torch.tensor([[[0.2, 1, 0, 0.4], [-1, 0, -0.1, -0.1], [0.1, 0, -1, -0.5], [0.4, -0.7, -0.5, 1]]])
# avg_pool2d = AvgPool2d(2, stride=2)
# print("The result of average pooling with 2-by-2 kernel and 2 stride is:\n", avg_pool2d(x))

# b. max pooling with a pooling region of 2-by-2 and stride = 2
# max_pool2d = MaxPool2d(2, stride=2)
# print("The result of max pooling with 2-by-2 kernel and 2 stride is:\n", max_pool2d(x))

# c. max pooling with a pooling region of 3-by-3 and stride = 1
# max_pool2d_ = MaxPool2d(3, stride=1)
# print("The result of max pooling with 3-by-3 kernel and 1 stride is:\n", max_pool2d_(x))


# Tutorial 5.9, 5.10 size of output
# 5.10 sequence of layers: 每一层的输出作为下一层的输入
# dim_feature_map = [98, 98]
# dim_mask = [4, 4]
# padding = 1
# stride = 2
# num_masks = 20
# outputHeight = 1 + (dim_feature_map[0] - dim_mask[0] + 2 * padding) / stride
# outputWidth = 1 + (dim_feature_map[1] - dim_mask[1] + 2 * padding) / stride
# print("The output size is:", (outputHeight, outputWidth, num_masks))
# len_feature_vec = outputHeight * outputWidth * num_masks
# print("After flattening, length of feature vector is:", len_feature_vec)


# Tutorial 6.3
# a. compute V(D, G)

# real samples
# x1 = np.transpose(np.array([1, 2]))
# x2 = np.transpose(np.array([3, 4]))

# fake samples
# x_1 = np.transpose(np.array([5, 6]))
# x_2 = np.transpose(np.array([7, 8]))

# theta_d1, theta_d2 = 0.1, 0.2

# real and fake samples have equal probability to be selected; 0.5
# E1 = 0.5 * math.log(pow(1 + pow(math.e, -(theta_d1 * x1[0] - theta_d2 * x1[1] - 2)), -1), math.e) + \
#    0.5 * math.log(pow(1 + pow(math.e, -(theta_d1 * x2[0] - theta_d2 * x2[1] - 2)), -1), math.e)
# E2 = 0.5 * math.log(1 - pow(1 + pow(math.e, -(theta_d1 * x_1[0] - theta_d2 * x_1[1] - 2)), -1), math.e) + \
#    0.5 * math.log(1 - pow(1 + pow(math.e, -(theta_d1 * x_2[0] - theta_d2 * x_2[1] - 2)), -1), math.e)

# V_D_G = E1 + E2
# print("The cost V(D, G) is:", V_D_G)

# b. determine the updated theta values
# lr = 0.02
# m = 2  # number of real samples/generated samples
# x_real = np.array([x1, x2])
# x_fake = np.array([x_1, x_2])
# alpha1_1 = (x_real[0][0] * pow(math.e, - (theta_d1 * x_real[0][0] - theta_d2 * x_real[0][1] - 2))) / \
#    (1 + pow(math.e, -(theta_d1 * x_real[0][0] - theta_d2 * x_real[0][1] - 2)))
# beta1_1 = -(x_fake[0][0] * pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2))) / \
#          (pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2)) * (pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2)) + 1))
# temp1 = alpha1_1 + beta1_1
# alpha2_1 = -(x_real[0][1] * pow(math.e, - (theta_d1 * x_real[0][0] - theta_d2 * x_real[0][1] - 2))) / \
#    (1 + pow(math.e, -(theta_d1 * x_real[0][0] - theta_d2 * x_real[0][1] - 2)))
# beta2_1 = (x_fake[0][1] * pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2))) / \
#          (pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2)) * (pow(math.e, -(theta_d1 * x_fake[0][0] - theta_d2 * x_fake[0][1] - 2)) + 1))
# temp2 = alpha2_1 + beta2_1
# alpha1_2 = (x_real[1][0] * pow(math.e, - (theta_d1 * x_real[1][0] - theta_d2 * x_real[1][1] - 2))) / \
#    (1 + pow(math.e, -(theta_d1 * x_real[1][0] - theta_d2 * x_real[1][1] - 2)))
# beta1_2 = (-x_fake[1][0] * pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2))) / \
#          (pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2)) * (pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2)) + 1))
# temp3 = alpha1_2 + beta1_2
# alpha2_2 = -(x_real[1][1] * pow(math.e, -(theta_d1 * x_real[1][0] - theta_d2 * x_real[1][1] - 2))) / \
#    (1 + pow(math.e, -(theta_d1 * x_real[1][0] - theta_d2 * x_real[1][1] - 2)))
# beta2_2 = (x_fake[1][1] * pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2))) / \
#          (pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2)) * (pow(math.e, -(theta_d1 * x_fake[1][0] - theta_d2 * x_fake[1][1] - 2)) + 1))
# temp4 = alpha2_2 + beta2_2
# change = (1 / m) * (np.array([temp1, temp2]) + np.array([temp3, temp4]))
# updated_theta = np.array([theta_d1, theta_d2]) + lr * change
# print("The updated theta value is:\n", updated_theta)


# Tutorial 7.4, 7.6 Karhunen-Loeve Transform KLT
# x = np.array([[0, 1],
#              [3, 5],
#              [5, 4],
#              [5, 6],
#              [8, 7],
#              [9, 7]])
# num_samples, num_features = x.shape
# avg = np.array([np.mean(x[:, i]) for i in range(num_features)])
# x_norm = x - avg  # zero-mean
# print("The equivalent zero-mean data is:\n", x_norm)
# cov_matrix = np.dot(np.transpose(x_norm), x_norm) / len(x)
# print("The covariance matrix is:\n", cov_matrix)
# eig_val, eig_vec = np.linalg.eig(cov_matrix)
# print("The eigenvectors and eigenvalues are:\n", eig_vec, '\n', '\n', eig_val)
# eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(num_features)]
# eig_pairs.sort(reverse=True)

# select the top k eigenvectors
# k = 1  # project onto the first self-defined number of principal components
# feature = np.array([ele[1] for ele in eig_pairs[:k]])
# get transformed data
# data = np.dot(x_norm, np.transpose(feature))
# print("The new data is:\n", data)  # either or both columns can be multiplied by -1


# Tutorial 7.7 Oja's learning rule
# lr = 0.01
# w = np.array([-1.0, 0.0])
# epoch = 2
# for i in range(epoch):
#    temp = []
#    for j in range(len(x_norm)):
#        y = np.dot(w, x_norm[j].T)
#        temp.append(lr * y * (x_norm[j] - y * w))
#    total = 0
#    for k in range(len(temp)):
#        total += temp[k]
#    w += np.array(total)
#    print("Epoch {0}: w = {1}".format(i+1, w))

# the 1st principal component using w calculated by Oja
# print("Project zero-mean data onto the first principal component is:", np.dot(w, x_norm.T))


# Tutorial 7.10 Linear Discriminant Analysis Fisher's method
# w1 = np.array([-1, 5])
# w2 = np.array([2, -6])
# x = np.array([[1, 2],
#              [2, 1],
#              [3, 3],
#              [6, 5],
#              [7, 8]])
# x1 = x[:3]  # class 1
# x2 = x[3:]  # class 2
# num_class1, num_feature_class1 = x1.shape
# mean_class1 = np.array([np.mean(x1[:, i]) for i in range(num_feature_class1)])
# num_class2, num_feature_class2 = x2.shape
# mean_class2 = np.array([np.mean(x2[:, i]) for i in range(num_feature_class2)])
# sb1 = np.abs(np.dot(w1, np.transpose(mean_class1 - mean_class2))) ** 2
# sb2 = np.abs(np.dot(w2, np.transpose(mean_class1 - mean_class2))) ** 2
# count1_1, count1_2, count2_1, count2_2 = 0.0, 0.0, 0.0, 0.0
# for i in range(len(x1)):
#    count1_1 += np.dot(w1, np.transpose(x1[i] - mean_class1)) ** 2
#    count2_1 += np.dot(w2, np.transpose(x1[i] - mean_class1)) ** 2
# for i in range(len(x2)):
#    count1_2 += np.dot(w1, np.transpose(x2[i] - mean_class2)) ** 2
#    count2_2 += np.dot(w2, np.transpose(x2[i] - mean_class2)) ** 2
# sw1 = count1_1 + count1_2
# sw2 = count2_1 + count2_2
# J_w1 = sb1 / sw1
# J_w2 = sb2 / sw2
# print("The values of cost function J(w) for w1 and w2 are:", J_w1, J_w2)
# if J_w1 > J_w2:
#    print("w1 is a more effective projection weight.")
# elif J_w1 < J_w2:
#    print("w2 is a more effective projection weight.")
# else:
#    print("w1 and w2 are equally effective weights.")

# Tutorial 7.11 Extreme Learning Machine
# w = np.array([0, 0, 0, -1, 0, 0, 2])
# x = np.array([[0, 0],
#              [0, 1],
#              [1, 0],
#              [1, 1]])
# X = []
# for i in range(len(x)):
#    X.append(np.insert(x[i], 0, 1))
# X = np.array(X).reshape(4, 3).T
# V = np.array([[-0.62, 0.44, -0.91],
#              [-0.81, -0.09, 0.02],
#              [0.74, -0.91, -0.60],
#              [-0.82, -0.92, 0.71],
#              [-0.26, 0.68, 0.15],
#              [0.80, -0.94, -0.83]])
# H = np.dot(V, X)
# H[H <= 0], H[H > 0] = 0, 1
# height, width = H.shape
# Y = []
# for i in range(width):
#    Y.append(np.insert(H[:, i], 0, 1))
# Y = np.array(Y).T
# Z = np.dot(w, Y)
# print("The response of the output neuron to all input patterns is:", Z)

# Tutorial 7.12, 7.13 Sparse Coding
# sparsity is measured as the count of elements that are non-zero
# prefer sparser (less non-zero elements)

# y1 = np.array([1, 2, 0, -1])
# y2 = np.array([0, 0.5, 1, 0])
# V = np.array([[1, 1, 2, 1],
#              [-4, 3, 2, -1]])
# x = np.array([2, 3])
# a = 1  # trade-off parameter
# sparsity1 = np.count_nonzero(y1)
# sparsity2 = np.count_nonzero(y2)
# print("The sparsity of these two alternatives is:", sparsity1, sparsity2)  # 选较小的值
# error1 = np.linalg.norm(x.T - np.dot(V, y1.T))
# cost1 = np.linalg.norm(x.T - np.dot(V, y1.T)) + a * np.count_nonzero(y1)
# error2 = np.linalg.norm(x.T - np.dot(V, y2.T))
# cost2 = np.linalg.norm(x.T - np.dot(V, y2.T)) + a * np.count_nonzero(y2)
# print("The reconstruction errors for these two alternatives are:", error1, error2)  # 选较小的值
# print("The costs for these two alternatives are:", cost1, cost2)  # 选较小的值

# Feature Extraction
# w = np.array([5, -3])
# X = np.array([[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]])
# Y = np.dot(w, X.T)
# print("The coordinates in the new feature space are:", Y)


# Tutorial 8 SVM
# def svm(x, y):
#    clf = SVC(kernel='linear')
#    clf.fit(x, y)
#    print("w and w0 of the SVM classifier is:", clf.coef_, clf.intercept_)
#    print("The support vectors are:\n", clf.support_vectors_)  # 支持向量不一定对，自己画图验证一下
#    margin = 2 / np.linalg.norm(clf.coef_)
#    print("The margin given by the hyperplane is:", margin)


# feature mapping function
# X = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2], [1, 1], [1, -1], [-1, -1], [-1, 1]])
# Y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
# for data in X:
#    if np.linalg.norm(data) > 2:
#        temp = data[0].copy()
#        data[0] = 4 - (data[1] / 2) + abs(data[0] - data[1])
#        data[1] = 4 - (temp / 2) + abs(temp - data[1])
#    else:
#        data[0] -= 2
#        data[1] -= 3

# svm(X, Y)

# Tutorial 10 Q1 K-Means Clustering
# def kmeans(x, m1, m2, m1_list, m2_list):
#    while True:
#        Class = []
#        for i in range(len(x)):
#            dist1 = np.linalg.norm(x[i] - m1)
#            dist2 = np.linalg.norm(x[i] - m2)
#            if dist2 == min(dist1, dist2):
#                Class.append(2)
#            elif dist1 == min(dist1, dist2):
#                Class.append(1)
#        c1_data = []
#        c2_data = []
#        for i in range(len(Class)):
#            if Class[i] == 1:
#                c1_data.append(x[i])
#            elif Class[i] == 2:
#                c2_data.append(x[i])
#        c1_data = np.array(c1_data)
#        m1 = np.mean(c1_data, axis=0)
#        m1_list.append(m1)
#        c2_data = np.array(c2_data)
#        m2 = np.mean(c2_data, axis=0)
#        m2_list.append(m2)
#        now_index = len(m1_list) - 1
#        print("Iteration {0}: m1 = {1}, m2 = {2}".format(now_index, m1_list[now_index], m2_list[now_index]))
#        if (m1_list[now_index] == m1_list[now_index - 1]).all():
#            if (m2_list[now_index] == m2_list[now_index - 1]).all():
#                break
#    print("The class for these samples are:", Class)


# X = np.array([[-1, 3],
#              [1, 2],
#              [0, 1],
#              [4, 0],
#              [5, 4],
#              [3, 2]])
# m1, m2 = np.array([-1, 3]), np.array([3, 2])
# m1_list, m2_list = [m1], [m2]
# kmeans(X, m1, m2, m1_list, m2_list)


# Tutorial 10 Q6 Agglomerative Hierarchical Clustering
# X = np.array([[-1, 3],
#              [1, 2],
#              [0, 1],
#              [4, 0],
#              [5, 4],
#              [3, 2]])

# n_clusters: target number of clusters; linkage: ward, single, average, complete
# clustering = AgglomerativeClustering(n_clusters=3, linkage='average').fit(X)
# print("The assigned labels of these samples are:", clustering.labels_)

# Tutorial 10 Q2 PCA
# a., b.
# X = np.array([[-4.6, -3],
#              [-2.6, 0],
#              [-0.6, -1],
#              [2.4, 2],
#              [5.4, 2]])
# pca_2d = PCA(n_components=2)
# X_2d = pca_2d.fit_transform(X)
# pca_1d = PCA(n_components=1)
# X_1d = pca_1d.fit_transform(X)
# means_2d = KMeans(n_clusters=2).fit(X_2d)
# means_1d = KMeans(n_clusters=2).fit(X_1d)
# new_data = np.array([[3, -2, 5], [3, -2, 5]])
# new_data_2d = pca_2d.fit_transform(new_data)
# new_data_1d = pca_1d.fit_transform(new_data)
# print("Projecting the data onto 2D plane:\n", X_2d)
# print("Projecting the data onto 1D plane:\n", X_1d)
# print("The assigned labels of the transformed 2D samples are:", means_2d.labels_)
# print("The new data based on the transformed 2D samples belongs to:", means_2d.predict(new_data_2d))
# print("The assigned labels of the transformed 1D samples are:", means_1d.labels_)
# print("The new data based on the transformed 1D samples belong to:", means_1d.predict(new_data_1d))

# Tutorial 10 Q3 Competitive learning (without normalisation)
# a. Find the cluster centres for each iteration
# X = np.array([[1.0, 0.0],
#              [0.0, 2.0],
#              [1.0, 3.0],
#              [3.0, 0.0],
#              [3.0, 1.0]])
# m1, m2 = np.array([1.0, 1.0]), np.array([2.0, 2.0])  # initial cluster centres
# m = [m1, m2]
# m1, m2, m3 = X[0] / 2, X[2] / 2, X[4] / 2  # initial cluster centres
# m = [m1, m2, m3]
# lr = 0.5
# samples = np.array([X[2], X[0], X[0], X[4], X[5]])  # training samples
# samples = X.copy()  # training samples
# for i in range(len(samples)):
#    dist = []
#    for j in range(len(m)):
#        dist.append(np.linalg.norm(samples[i] - m[j]))
#    min_index = np.argmin(dist)
#    m[min_index] += lr * (samples[i] - m[min_index])
#    print("Iteration {0}: m_{1} = {2}".format(i+1, min_index+1, m[min_index]))
# print("After {0} iterations, the cluster centres are {1}:".format(len(samples), m))


# b., c. classify the samples and the new data
# def classify(X, m):
#    new_data = np.array([0, -2])
#    dist_new = []
#    classes = []
#    for i in range(len(X)):
#        dist = []
#        for j in range(len(m)):
#            dist.append(np.linalg.norm(X[i]-m[j]))
#            dist_new.append(np.linalg.norm(new_data-m[j]))
#        classes.append(np.argmin(dist) + 1)
#    print("The classes of these samples are:", classes)
#    print("The class of the new data is:", np.argmin(dist_new) + 1)


# classify(X, m)


# Tutorial 10 Q4 Basic leader follower algorithm (without normalisation)
# a. Find the cluster centres for each iteration
# m = []
# theta = 3
# lr = 0.5
# m.append(samples[0])
# for i in range(len(samples)):
#    dist = []
#    for j in range(len(m)):
#        dist.append(np.linalg.norm(samples[i] - m[j]))
#    min_index = np.argmin(dist)
#    if np.linalg.norm(samples[i] - m[min_index]) < theta:
#        m[min_index] += lr * (samples[i] - m[min_index])
#    else:
#        m.append(samples[i])
#    print("Iteration {0}: m = {1}".format(i+1, m))

# b., c. classify the samples and the new data
# classify(X, m)


# Tutorial 10 Q5 Fuzzy K-means algorithm
# X = np.array([[-1.0, 3.0],
#              [1.0, 2.0],
#              [0.0, 1.0],
#              [4.0, 0.0],
#              [5.0, 4.0],
#              [3.0, 2.0]])
# b = 2
# u = np.array([[1, 0.5, 0.5, 0.5, 0.5, 0], [0, 0.5, 0.5, 0.5, 0.5, 1]])
# m1 = sum([(u[0][i] ** 2) * X[i] for i in range(len(X))]) / (sum([u[0][i] ** 2 for i in range(len(X))]))
# m2 = sum([(u[1][i] ** 2) * X[i] for i in range(len(X))]) / (sum([u[1][i] ** 2 for i in range(len(X))]))
# m1_list = [m1]
# m2_list = [m2]
# print("Iteration 1: m1 = {0}, m2 = {1}".format(m1, m2))
# while True:
#    for i in range(len(X)):
#        u[0][i] = ((1 / np.linalg.norm(X[i] - m1)) ** (2 / (b-1))) / (((1 / np.linalg.norm(X[i] - m1)) ** (2 / (b-1))) + ((1 / np.linalg.norm(X[i] - m2)) ** (2 / (b-1))))
#        u[1][i] = ((1 / np.linalg.norm(X[i] - m2)) ** (2 / (b-1))) / (((1 / np.linalg.norm(X[i] - m1)) ** (2 / (b-1))) + ((1 / np.linalg.norm(X[i] - m2)) ** (2 / (b-1))))
#    m1 = sum([(u[0][i] ** 2) * X[i] for i in range(len(X))]) / (sum([u[0][i] ** 2 for i in range(len(X))]))
#    m2 = sum([(u[1][i] ** 2) * X[i] for i in range(len(X))]) / (sum([u[1][i] ** 2 for i in range(len(X))]))
#    m1_list.append(m1)
#    m2_list.append(m2)
#    current_iter = len(m1_list) - 1
#    print("Iteration {0}: m1 = {1}, m2 = {2}".format(current_iter+1, m1_list[current_iter], m2_list[current_iter]))

    # terminate when both coordinates of both clusters centres change by less than 0.5
#    if (m1_list[current_iter] - m1_list[current_iter - 1] < 0.5).all():
#        if (m2_list[current_iter] - m2_list[current_iter - 1] < 0.5).all():
#            print("After {0} iterations, the algorithm converges.".format(current_iter+1))
#            print("m1 = {0}, m2 = {1}".format(m1_list[current_iter], m2_list[current_iter]))
#            break
