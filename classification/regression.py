import numpy as np


def l1_norm_subderivative(gradient):
    d = gradient.shape[1]
    temp = np.zeros((1, d))
    for j in range(d):
        if gradient[0, j] > 0:
            temp[0, j] = 1
        elif gradient[0, j] < 0:
            temp[0, j] = -1
        else:
            temp[0, j] = 0
    return temp


def lr_iteration(gradient, data, label, lamda=0.01, gamma=1):
    delta = (1/(1 + np.math.exp(-1 * label * np.dot(gradient, data.T))) - 1) * label * data + lamda * l1_norm_subderivative(gradient)
    return gradient - gamma * delta


def compute_loss(gradient, data, label, method, lamda=0.01):
    n = data.shape[0]
    loss = 0
    for i in range(n):
        if method == 'logistic':
            loss += np.math.log(1 + np.math.exp(-1 * label[i] * gradient * data[i].T)) + lamda * np.linalg.norm(gradient)
        elif method == 'ridge':
            loss += float((label[i] - gradient * data[i].T) ** 2 + lamda * np.linalg.norm(gradient, 2))
    return loss / n


def ridge_iteration(gradient, data, label, lamda=0.01, alpha=1):
    temp = gradient * data.T
    delta = -2 * (label - temp) * data + 2 * lamda * gradient
    return gradient - alpha * delta


