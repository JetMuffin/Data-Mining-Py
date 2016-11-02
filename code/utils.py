import numpy as np

import time
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1 - t0))
               )
        return result

    return function_timer


def read_data(file):
    with open(file) as f:
        data = np.loadtxt(f, delimiter=',', dtype=float)
        return np.mat(data[:, :-1]).T, np.mat(data[:, -1]).T


def l2_norm_distance(data):
    d, n = data.shape
    dis = np.zeros((n, n))
    for i in range(n):
        diff = np.tile(data[:, i], (1, n)) - data
        dis[:, i] = np.sqrt(np.sum(np.square(diff), axis=0))
    return dis


def l1_norm_distance(data):
    d, n = data.shape
    dis = np.zeros((n, n))
    for i in range(n):
        diff = np.tile(data[:, i], (1, n)) - data
        dis[:, i] = np.sum(np.abs(diff), axis=0)
    return dis


def knn_graph(dis, k):
    n, n = dis.shape
    graph = np.zeros(dis.shape)
    sorted_index = np.argsort(dis, axis=1)[:, :(k+1)]
    for i in range(n):
        graph[i, sorted_index[i, 1:]] = 1
        graph[sorted_index[i, 1:], i] = 1
    return graph


def pca(data, k):
    n = data.shape[1]
    data = data - data.mean(axis=1)

    # calculate k largest eigenvectors of covariance matrix
    eig_vals, eig_vectors = np.linalg.eig(data * data.T / n)
    eig_index = np.argsort(eig_vals)[-1:-(k+1):-1]
    eig_vectors = eig_vectors[:, eig_index]

    return eig_vectors.T


def cal_purity_and_gini(cluster, labels, k):
    label_map = {}
    count = 0
    gini = 0.0

    for i in range(len(labels)):
        cls = labels[i, 0]
        label_map[cls] = label_map.get(cls, 0) + 1

    for i in range(k):
        cluster_map = {}
        cluster_gini = 0.0

        for j in range(cluster[i].shape[0]):
            cls = labels[cluster[i][j], 0]
            cluster_map[cls] = cluster_map.get(cls, 0) + 1

        max_cls_count = max(cluster_map.items(), key=lambda x:x[1])[1]
        count += max_cls_count

        for cls, cls_count in cluster_map.items():
            cluster_gini += (cls_count / float(cluster[i].shape[0])) **2
        cluster_gini = 1 - cluster_gini
        gini += cluster_gini * cluster[i].shape[0]

    return float(count) / float(len(labels)), gini / float(len(labels))


def show(data, k, centroids, cluster):
    from matplotlib import pyplot as plt
    data = pca(data, 2) * data
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'w', 'y', '#cc5500', '#ff8888']
    for i in xrange(k):
        for j in xrange(cluster[i].shape[0]):
            plt.plot(data[0, cluster[i][j]], data[1, cluster[i][j]], marker='o', color=colors[i])
    for i in range(k):
        plt.plot(data[0, centroids[i]], data[1, centroids[i]], color=colors[i], marker='D', markersize=12)
    plt.show()
