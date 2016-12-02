import random

import numpy as np


def load_matrix_from_txt(filename):
    with open(filename) as f:
        type_line = f.readline()
        types = np.mat(map(lambda x: int(x.strip()), type_line.split(','))).T

        data = np.loadtxt(f, delimiter=',')
        features = np.mat(np.copy(data[:, :-1]))
        labels = np.mat(np.copy(data[:, -1])).T
        labels[labels == 0] = -1

        return features, labels, types


def compute_accuracy(predict, label):
    size = label.shape[0]
    true = len(np.where(predict == label)[0])

    return 1.0 * true / size


def k_fold_split(k, data, label):
    size = data.shape[0]
    x = list(range(size))
    random.shuffle(x)
    splices = np.array_split(x, k)

    cross_validation_splices = []
    for i in range(k):
        train_splices = []
        for j in range(k):
            if j != i:
                train_splices.extend(splices[i])
        cross_validation_splices.append((train_splices, splices[i]))

    return cross_validation_splices


def cross_validation(features, labels, fold=10):
    n = features.shape[0]
    x = list(range(n))
    # random.shuffle(x)
    train_features, test_features = features[x[n // fold: n]], features[x[0: n // fold]]
    train_labels, test_labels = labels[x[n // fold: n]], labels[x[0: n // fold]]

    return train_features, train_labels, test_features, test_labels
