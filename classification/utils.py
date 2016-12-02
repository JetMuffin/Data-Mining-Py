import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab


def load_matrix_from_txt(filename):
    with open(filename) as f:
        data = np.loadtxt(f, delimiter=',')

        # random shuffle data
        np.random.shuffle(data)

        # add bias column
        n, d = data.shape
        label = np.copy(data[:, -1].T)
        data[:, -1] = np.ones((1, n))
        train = np.mat(data)

        return train, label


def random_list(n):
    list = [i for i in range(n)]
    random.shuffle(list)
    return list


def draw(train_errors, test_errors, loss, filename, max_iteration, step, lamda, gamma):
    x = [i for i in range(len(train_errors))]

    plt.figure(1)
    plt.plot(x, loss)
    plt.title('Ridge Regression (lambda = ' + str(lamda) + ', learning rate = ' + str(gamma) + ')')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    pylab.savefig(filename + '-loss.png')

    plt.figure(2)
    plt.plot(x, train_errors, 'b', label='train errors')
    plt.plot(x, test_errors, 'r', label='test errors')
    plt.title('Ridge Regression (lambda = ' + str(lamda) + ', learning rate = ' + str(gamma) + ')')
    plt.xlabel('iterations')
    plt.ylabel('error rate')
    plt.legend()

    plt.show()
    pylab.savefig(filename + '-error.png')