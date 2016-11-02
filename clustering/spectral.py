import os
import sys
from kmediods import kmedoid
from utils import *
import scipy.sparse.linalg as linalg


def spectral(file, dis, k_neighbor, k_mediod):
    distance_matrix_file = "{}.distance_matrix.{}".format(file, k_neighbor)

    if os.path.exists(distance_matrix_file):
        print "Distance matrix file exists, load from file {}".format(distance_matrix_file)
        dis = np.load(distance_matrix_file)
    else:
        W = knn_graph(dis, k_neighbor)
        D = np.mat(np.diag(np.sum(W, axis=0)))
        # L_norm = np.identity(dis.shape[0]) - np.sqrt(D.I) * W * np.sqrt(D.I)
        L_norm = D - W

        eig_vals, eig_vectors = linalg.eigsh(L_norm, k=k_mediod+1, which='SA')
        eig_vectors = np.mat(eig_vectors[:, 1:]).T

        dis = l1_norm_distance(eig_vectors)
        dis.dump(distance_matrix_file)

    cluster, centroids = kmedoid(dis, k_mediod, 100)
    return cluster, centroids


def cluster(file, k, display=False):
    distance_file = "{}_l2_distance".format(file)
    feature, labels = read_data(file)

    if not os.path.exists(distance_file):
        print "Distance file not found, compute it and save to file."
        dis = l2_norm_distance(feature)
        dis.dump(distance_file)
    else:
        print "Load distance matrix from {}".format(distance_file)
        dis = np.load(distance_file)

    for k_neighbor in [3, 6, 9]:

        max_purity = 0
        max_gini_index = 0
        for t in range(10):
            cluster, centroids = spectral(file, dis, k_neighbor, k)
            purity, gini_index = cal_purity_and_gini(cluster, labels, k)
            print "k_neighbor={}, {} trys: {}/{}".format(k_neighbor, t, purity, gini_index)

            if purity > max_purity:
                max_purity = purity
                max_gini_index = gini_index

        print "Max purity and gini index: {}/{}".format(max_purity, max_gini_index)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python kmedoids <data> <k>"
        sys.exit(1)

    cluster(sys.argv[1], int(sys.argv[2]))
