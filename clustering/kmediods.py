import os
from utils import *
import sys


def kmedoid(dis, k, max_iteration):
    n, n = dis.shape

    # select initial centroid randomly
    centroid_index = np.sort(np.random.choice(n, k))
    centroid_index_copy = np.copy(centroid_index)

    cluster = {}

    for t in xrange(max_iteration):

        # determine which cluster each point belongs to
        labels = np.argmin(dis[:, centroid_index], axis=1)
        for i in range(k):
            cluster[i] = np.where(labels == i)[0]

        # calculate new cluster mediods
        for i in range(k):
            cluster_dis = np.mean(dis[np.ix_(cluster[i], cluster[i])], axis=1)
            cluster_index = np.argmin(cluster_dis)
            centroid_index_copy[i] = cluster[i][cluster_index]
        np.sort(centroid_index_copy)

        # check if it has been converged
        if np.array_equal(centroid_index, centroid_index_copy):
            break

        centroid_index = np.copy(centroid_index_copy)
    else:
        labels = np.argmin(dis[:, centroid_index], axis=1)
        for i in range(k):
            cluster[i] = np.where(labels == i)[0]

    return cluster, centroid_index


def cluster(file, k, display=False):
    distance_file = "{}_l1_distance".format(file)
    feature, labels = read_data(file)

    if not os.path.exists(distance_file):
        print "Distance file not found, compute it and save to file."
        dis = l1_norm_distance(feature)
        dis.dump(distance_file)
    else:
        print "Load distance matrix from {}".format(distance_file)
        dis = np.load(distance_file)

    max_purity = 0
    max_gini_index = 0.0
    for t in range(10):
        cluster, centroids = kmedoid(dis, k, 100)

        purity, gini_index = cal_purity_and_gini(cluster, labels, k)
        print "Purity of {} trys: {}".format(t, purity)
        print "Gini index of {} trys: {}".format(t, gini_index)

        if purity > max_purity:
            max_purity = purity
            max_gini_index = gini_index

    print "The max purity of 10 trys is: {}".format(max_purity)
    print "The max gini index of 10 trys is: {}".format(max_gini_index)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python kmedoids <data> <k>"
        sys.exit(1)

    cluster(sys.argv[1], int(sys.argv[2]))

