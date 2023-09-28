import numpy as np
import math
from matplotlib import pyplot as plt
import time
import gradient_descent as gd

"""----------------------计算归一化后的损失函数----------------------"""


def h(a, b):
    return 0.297999 + 0.333333 * a ** 2 - 1 * b + 1 * b ** 2 + a * (-0.398679 + 1 * b)


"""----------------------单变量函数拟合程序----------------------"""


def regression(method):
    """----------------------数据集初始化----------------------"""
    # x, y分别为函数自变量和因变量
    # 对x, w进行增广
    x = np.linspace(-4, 4, 8001)
    y = np.zeros(8001)
    for i in range(0, len(x)):
        y[i] = x[i] * math.cos(0.25 * math.pi * x[i])
    temp = np.ones((1, 8001))
    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    temp = temp.T
    x = np.hstack([x, temp])

    """----------------------代码运行----------------------"""
    w = []
    hist = []
    eta = 0.4
    max_iter = 10
    batch_size = 500
    epsilon = 1e-6
    alpha_rms_drop = 0.9
    lamda = 0.9
    alpha_adam = 0.001
    beta1 = 0.9
    beta2 = 0.999
    time_start = time.time()
    match method:
        case 'naive':
            w, iteration, loss_f, hist = gd.gradient_descent(x, y, eta, max_iter)
        case 'stochastic':
            w, iteration, loss_f, hist = gd.sgd(x, y, eta, max_iter, batch_size)
        case 'adagrad':
            w, iteration, loss_f, hist = gd.adagrad(x, y, eta, max_iter, epsilon)
        case 'rms_drop':
            w, iteration, loss_f, hist = gd.rms_drop(x, y, eta, max_iter, epsilon, alpha_rms_drop)
        case 'momentum':
            w, iteration, loss_f, hist = gd.momentum(x, y, eta, max_iter, lamda)
        case 'adam':
            w, iteration, loss_f, hist = gd.adam(x, y, eta, max_iter, alpha_adam, beta1, beta2, epsilon)
    time_end = time.time()
    time_spend = time_end - time_start

    print("w=", w)
    print("算法运行时间=", time_spend, "s")

    """----------------------绘图----------------------"""
    w = np.array(w)
    w = w.T
    plt.figure("函数图像")
    plt.plot(x[:, 0], y, c='b', label="y=x*cos(0.25Pi*x)")
    y_fit = np.dot(x, w)
    w_approx = np.around(w, 3)
    str1 = 'y=(' + str(w_approx[0][0]) + ')x+(' + str(w_approx[1][0]) + ')'
    plt.plot(x[:, 0], y_fit, c='r', label=str1)
    plt.xlim(-5, 5)
    plt.ylim(-6, 6)
    plt.legend()

    plt.figure("损失函数等高图")
    hist = np.array(hist)
    a = np.arange(min([-0.7, min(hist[:, 0])]), max(hist[:, 0]) + 0.1, 0.01)
    b = np.arange(min(hist[:, 1]) - 0.1, max([0.9, max(hist[:, 1])]), 0.01)
    A, B = np.meshgrid(a, b)
    plt.contourf(A, B, h(A, B), 20, cmap=plt.get_cmap('coolwarm'))
    plt.scatter(hist[:, 0], hist[:, 1], c='r')
    plt.plot(hist[:, 0], hist[:, 1], c='r', label=method)
    plt.legend()

    plt.show()

    return 0
