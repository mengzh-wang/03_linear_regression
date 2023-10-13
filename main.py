import numpy as np
from matplotlib import pyplot as plt
import classification
import function_regression

"""线性回归和梯度下降用于二分类"""
classification.sort()

"""
梯度下降用于求回归方程
method参数用于选则不同的梯度下降算法，可选naive, stochatic, adagrad, rms_drop, momentum, adam
>>>>>>>>>>需要在当前文件夹新建‘figs’文件夹，否则图片无法正常输出。<<<<<<<<<<
"""

hist_all = np.zeros((1, 2))
w_all = np.zeros((1, 2))
method = ['naive', 'stochastic', 'adagrad', 'rms_drop', 'momentum', 'adam']
iteration = 0
for i in range(6):
    hist, w, iteration = function_regression.regression(method[i])
    hist_all = np.vstack((hist_all, hist))
    w_all = np.vstack((w_all, w.T))
hist_all = np.delete(hist_all, 0, 0)
w_all = np.delete(w_all, 0, 0)
plt.figure("梯度下降过程（对比）")
a = np.arange(min([-0.7, min(hist_all[:, 0]) - 0.1]), max(hist_all[:, 0]) + 0.1, 0.01)
b = np.arange(min(hist_all[:, 1]) - 0.1, max([0.9, max(hist_all[:, 1]) + 0.1]), 0.01)
A, B = np.meshgrid(a, b)
plt.contourf(A, B, function_regression.h(A, B), 20, cmap=plt.get_cmap('coolwarm'))
length = iteration + 1
for i in range(6):
    plt.scatter(hist_all[length * i:length * (i + 1), 0], hist_all[length * i:length * (i + 1), 1])
    plt.plot(hist_all[length * i:length * (i + 1), 0], hist_all[length * i:length * (i + 1), 1], label=method[i])
plt.legend()
# plt.savefig('../figs/contour_all.png')
plt.close()
# plt.show()
