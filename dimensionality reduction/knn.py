import numpy as np


def neighbor_distances(data, data_set):
    diff = np.tile(data, (1, data_set.shape[1])) - data_set
    dist = np.sqrt(np.sum(np.square(diff), axis=0))
    sorted_index = np.argsort(dist, axis=1)
    return sorted_index, dist


def knn(data, features, labels, k):
    distances, _ = neighbor_distances(data, features)
    class_count = {}

    # count labels of k nearest neighbor
    for i in range(k):
        current_point = distances[0, i]
        current_class = labels[current_point, 0]
        class_count[current_class] = class_count.get(current_class, 0) + 1

    return max(class_count.items(), key=lambda x: x[1])[0]