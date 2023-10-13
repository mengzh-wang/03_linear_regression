import numpy as np
from matplotlib import pyplot as plt
import time
import gradient_descent as gd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""----------------------结果统计----------------------"""


def statistic(w, xin, yin, xout, yout):
    wrong_cases_train = 0
    wrong_cases_test = 0
    nin = len(xin)
    nout = len(xout)
    for j in range(nin):
        if np.dot(w, xin[j]) * yin[j] <= 0:
            wrong_cases_train += 1
    wrong_rate_train = wrong_cases_train / nin

    for j in range(nout):
        if np.dot(w, xout[j]) * yout[j] <= 0:
            wrong_cases_test += 1
    wrong_rate_test = wrong_cases_test / nout

    print("训练集正确率=", 1 - wrong_rate_train)
    print("测试集正确率=", 1 - wrong_rate_test)

    return 0


"""----------------------广义逆线性回归----------------------"""


def linear_regression(x, y):
    xt = x.T
    temp = np.dot(xt, x)
    temp = np.linalg.inv(temp)
    gen_inverse = np.dot(temp, xt)
    wtemp = np.dot(gen_inverse, y)
    w = wtemp.T
    w = w.tolist()
    return w


def sort():
    """----------------------数据集初始化----------------------"""

    # 数据分布与规模
    u1 = [-5, 0]
    s1 = [[2, 0], [0, 2]]
    u2 = [0, 5]
    s2 = [[2, 0], [0, 2]]
    n = 200
    train_rate = 0.8
    n_train = int(n * train_rate)
    n_test = n - n_train
    # 数据填充

    x1 = np.empty([n, 2])  # A
    x2 = np.empty([n, 2])  # B
    x_train = np.empty([n_train * 2, 2])  # 320
    x_test = np.empty([n_test * 2, 2])  # 80

    for i in range(n):  # 200
        x1[i] = np.random.multivariate_normal(u1, s1)
        x2[i] = np.random.multivariate_normal(u2, s2)

    for i in range(n_train):  # 160
        x_train[i] = x1[i]  # A
        x_train[n_train + i] = x2[i]  # B
    for i in range(n_test):  # 40
        x_test[i] = x1[i]  # A
        x_test[n_test + i] = x2[i]  # B

    aug1 = np.ones((n_train * 2, 1))
    x_train = np.hstack((x_train, aug1))
    aug2 = np.ones((n_test * 2, 1))
    x_test = np.hstack((x_test, aug2))

    y_train = np.empty([n_train * 2, 1])
    for i in range(n_train):
        y_train[i] = 1
        y_train[n_train + i] = -1
    y_test = np.empty([n_test * 2, 1])
    for i in range(n_test):
        y_test[i] = 1
        y_test[40 + i] = -1

    """----------------------代码运行----------------------"""

    time_lg_start = time.time()
    w_lg = linear_regression(x_train, y_train)
    time_lg_end = time.time()
    time_lg_spend = time_lg_end - time_lg_start

    time_gd_start = time.time()
    eta_gd = 0.5
    max_iter_gd = 50
    # w_gd, iteration, loss_f, hist = gd.gradient_descent(x_train, y_train, eta_gd, max_iter_gd)
    w_gd, iteration, loss_f, hist = gd.gradient_descent_epoch(x_train, y_train, eta_gd, 320, 20)
    time_gd_end = time.time()
    time_gd_spend = time_gd_end - time_gd_start
    loss_f = np.array(loss_f)

    x_min = min(min(x1[:, 0]), min(x2[:, 0]))
    x_max = max(max(x1[:, 0]), max(x2[:, 0]))
    y_min = min(min(x1[:, 1]), min(x2[:, 1]))
    y_max = max(max(x1[:, 1]), max(x2[:, 1]))
    x_co = np.linspace(x_min - 1, x_max + 1)

    print("--------------广义逆结果统计--------------")
    print("w=", w_lg)
    statistic(w_lg, x_train, y_train, x_test, y_test)
    print("算法运行时间=", time_lg_spend, "s")

    plt.figure("广义逆算法")
    str1 = "广义逆, x1~N(%s,%s), x2~N(%s,%s)" % (u1, s1, u2, s2)
    plt.title(str1)
    #z_pla = -(w_lg[0][0] / w_lg[0][1]) * x_co
    z_pla = -(w_lg[0][0] / w_lg[0][1]) * x_co - w_lg[0][2] / w_lg[0][1]
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='b')
    plt.plot(x_co, z_pla, c='g')
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    print("--------------梯度下降结果统计--------------")
    print("w=", w_gd)
    print("迭代次数=", iteration)
    print("损失函数=", loss_f[len(loss_f) - 1, 1])
    statistic(w_gd, x_train, y_train, x_test, y_test)
    print("算法运行时间=", time_gd_spend, "s")

    plt.figure("梯度下降算法")
    str2 = "梯度下降, x1~N(%s,%s), x2~N(%s,%s)" % (u1, s1, u2, s2)
    plt.title(str2)
    z_gd = -(w_gd[0][0] / w_gd[0][1]) * x_co - w_gd[0][2] / w_gd[0][1]
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='b')
    plt.plot(x_co, z_gd, c='g')
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    plt.figure("梯度下降损失函数")
    plt.title(str2)

    plt.plot(loss_f[:, 0], loss_f[:, 1], c='k')

    plt.show()

    return 0
