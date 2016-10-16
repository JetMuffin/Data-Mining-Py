import numpy as np
import knn


def svd(data, k):
    [U, s, V] = np.linalg.svd(data, full_matrices=False)
    U = U[:, :k]

    return U.T


def pca(data, k):
    n = data.shape[1]
    data = data - data.mean(axis=1)

    # calculate k largest eigenvectors of covariance matrix
    eig_vals, eig_vectors = np.linalg.eig(data * data.T / n)
    eig_index = np.argsort(eig_vals)[-1:-(k+1):-1]
    eig_vectors = eig_vectors[:, eig_index]

    return eig_vectors.T


def isomap(data, k_dimension, k_neighbor=7):
    n = data.shape[1]
    graph = np.full((n, n), float('inf'))

    # construct weighted graph
    for i in range(n):
        point = data[:, i]
        neighbors, distances = knn.neighbor_distances(point, data)
        for j in range(min(k_neighbor, n)):
            graph[neighbors[0, j], i] = graph[i, neighbors[0, j]] = distances[0, neighbors[0, j]]

    # compute shortest distance between all pairs of points
    dist = spfa(graph)

    # calculate the dot-product matrix
    j = np.eye(n, n) - np.ones((n, n)) / n
    s = -0.5 * j * dist * j

    # eigen decompose s
    eig_vals, eig_vectors = np.linalg.eig(s)
    eig_index = np.argsort(eig_vals)[-1:-(k_dimension+1):-1]
    eig_vals = eig_vals[eig_index]
    eig_vals_diag = np.mat(np.diag(eig_vals))
    eig_vectors = eig_vectors[:, eig_index]

    return (eig_vectors * np.sqrt(eig_vals_diag)).T


def floyd(graph):
    """
    Compute shortest distances between each pairs of points in graph with O(n^3) complexity
    """
    n = graph.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if graph[i, k] < float('inf') and graph[k, j] < float('inf') and graph[i, j] > graph[i, k] + graph[k, j]:
                    graph[i, j] = graph[i, k] + graph[k, j]

    return graph


def spfa(graph):
    """
    Compute shortest distances between each pairs of points in a graph with O(kVE) complexity.
    Note that k is much less then V in a sparse graph.
    """
    n = graph.shape[0]
    queue = []
    edges = [[] for i in range(n)]

    for i in range(n):
        for j in range(n):
            if graph[i, j] < float('inf'):
                edges[i].append(j)
                edges[j].append(i)

    dist = np.full((n, n), float('inf'))
    inqueue = [False] * n
    for src in range(n):
        queue.append(src)
        inqueue[src] = True
        dist[src, src] = 0

        # while not queue.empty():
        while len(queue):
            u = queue.pop(0)
            inqueue[u] = False

            for v in edges[u]:
                if dist[src, v] > dist[src, u] + graph[u, v]:
                    dist[src, v] = dist[src, u] + graph[u, v]
                    if not inqueue[v]:
                        inqueue[v] = True
                        queue.append(v)
    return dist
