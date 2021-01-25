import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


class Widrow_Hoff:  # Sequential Widrow Hoff
    def __init__(self, x, y, label, a, b, lr):  # define a constructor function
        self.x = np.array(x)  # self 指向对象本身，把特征向量x转换成矩阵，传递给self的属性x
        self.y = np.array(y)  # 把预测目标（标签）y转换成矩阵，传递给self的属性y
        self.label = label  # 把label变量传递给self的属性label
        self.a = np.array(a)  # 把a的初始值转换成矩阵，传递给self的属性a
        self.b = np.array(b)
        self.lr = lr  # learning rate

    def train(self, epoch):  # training function, epoch: 训练周期
        for i in range(epoch):  # i gets value from 0~epoch-1 integers 外层循环控制训练的周期数
            for j in range(len(self.x)):  # calculates the number of features in the iris 遍历iris数据集中的每一个特征
                if self.y[j] in self.label:  # y[j] is prediction target in iris (label)
                    input_x = np.transpose(np.insert(self.x[j], 0, 1))  # 如果y[j] (iris中的标签) 在指定的标签label内，则在特征向量x[j]最前面，即索引为0的位置插入1
                else:
                    input_x = np.transpose(np.insert(-self.x[j], 0, -1))  # 如果y[j] (iris中的标签) 不在指定的标签label内，则在特征向量x[j]最前面，即索引为0的位置插入-1。并将特征向量x[j]的所有值取反
                g_value = np.dot(self.a, input_x)  # calculate g(x)
                print("parameter g of {}-th iteration is {}\n".format(i*len(self.x)+j+1, g_value))  # 显示该轮迭代中g(x)的值
                if g_value != self.b[j]:  # 如果这两个值相等，作差的结果为0，则a不变
                    self.a = self.__update(j, g_value, input_x)  # 调用__update()方法更新a的值
                print("parameter a of {}-th iteration is {}\n".format(i*len(self.x)+j+1, self.a))  # 显示该轮迭代中更新后的a的值
        print("parameter a of {}-th iteration is {}\n".format(i*len(self.x)+j+1, self.a))  # 显示最后一轮迭代后a的值

    def __update(self, j, g_value, input_x):  # 继承函数train，可以用train函数里的变量和参数
        a_new = self.a + self.lr * (self.b[j] - g_value) * np.transpose(input_x)
        return a_new

    def compute_percentage(self):
        count = 0  # 统计符合要求的标签的数量
        for i in range(len(self.x)):
            input_x = np.transpose(np.insert(self.x[i], 0, 1))  # 在每个特征向量的最前面插入1并转置
            g_value = np.dot(self.a, input_x)  # 求g(x)的值 (若没有训练模型，即没有调用train函数，则a的值不变；训练了模型，则调用的是训练后的a的值)
            if g_value > 0:
                if self.y[i] == 0:
                    count += 1  # 标签为0并且g(x)>0，则符合要求的标签数量+1
            else:
                if self.y[i] == 1 or self.y[i] == 2:
                    count += 1  # g(x)<=0并且标签为1或2，符合要求的标签数量+1
        print("the percentage is {}\n".format(count/len(self.x)))  # 带有符合要求的标签的特征向量的数量/特征向量的总数


if __name__ == "__main__":
#    x = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
#    y = [1, 1, 1, -1, -1]
#    a = [-1.5, 5, -1]
#    b = [2, 2, 2, 2, 2]
#    lr = 0.2

#    x = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]
#    y = [1, 1, 1, -1, -1, -1]
#    label = [1]
#    a = [1.0, 0.0, 0.0]
#    b = [1.0, 1.5, 1.5, 1.5, 1.5, 1.0]
#    lr = 0.1

    iris = datasets.load_iris()  # 获取iris数据集并赋值给iris对象
    x = iris.data  # 获取iris中的特征（特征向量）
    y = iris.target  # 获取iris中的预测目标（标签）
    label = [0]
    a = [0.5, 0.5, -2.5, 1.5, -1.5]  # 定义初始参数的值
    b = [1 for _ in range(len(x))]  # 定义计算新的a的公式中的b
    lr = 0.01  # 学习速率

    model = Widrow_Hoff(x=x, y=y, label=label, a=a, b=b, lr=lr)  # 定义 Sequential Widrow Hoff 学习算法模型
#    model.train(epoch=2)  # 训练模型
    model.compute_percentage()  # 计算符合条件的百分比

# KNN
#    neigh = KNeighborsClassifier(n_neighbors=5)
#    neigh.fit(x, y)
#    print(neigh.predict([[7.0, 2.1, 4.2, 1.7],
#                         [6.3, 3.9, 2.5, 1.1],
#                         [7.1, 3.7, 6.9, 1.8],
#                         [4.9, 2.2, 4.5, 1.8],
#                         [5.2, 2.8, 5.0, 1.5]]))
