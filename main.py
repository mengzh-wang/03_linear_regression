import classification
import function_regression

"""线性回归和梯度下降用于二分类"""
classification.sort()

"""
梯度下降用于求回归方程
method参数用于选则不同的梯度下降算法，可选naive, sgd, adagrad, rms_drop, momentum, adam
"""
method='naive'
function_regression.regression(method)
