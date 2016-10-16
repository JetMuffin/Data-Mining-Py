import numpy as np
import reduction
from knn import knn


def read_data(file):
    with open(file) as f:
        data = np.loadtxt(f, delimiter=',', dtype=float)
        return np.mat(data[:, :-1]).T, np.mat(data[:, -1]).T


def reduce(reduction_method, train_data, test_data, k_dimension=-1):
    if k_dimension > 0:
        if reduction_method.__name__ == "isomap":
            train_data_length = train_data.shape[1]
            merge_data = np.column_stack((train_data, test_data))

            reduction_data = reduction_method(merge_data, k_dimension)
            train_data = reduction_data[:, :train_data_length]
            test_data = reduction_data[:, train_data_length:]
        else:
            projection = reduction_method(train_data, k_dimension)
            train_data = projection * train_data
            test_data = projection * test_data

    return train_data, test_data


def predict(reduction_method, train_features, train_labels, test_features, test_labels, k_dimension, **kwargs):
    train_input = kwargs.get("dataset", "")

    train_features, test_features = reduce(reduction_method, train_features, test_features, k_dimension)
    test_feature_length = test_features.shape[1]

    true_count = 0
    for i in range(test_feature_length):
        classify_result = knn(test_features[:, i], train_features, train_labels, 1)
        if classify_result == test_labels[i]:
            true_count += 1
    accuracy = float(true_count) / float(test_feature_length)

    print "method={}, input={}, k={}: {} ({}/{})".format(reduction_method.__name__, train_input, k_dimension,
                                                         accuracy, true_count, test_feature_length)


def main():
    for reduction_method in [reduction.pca, reduction.svd, reduction.isomap]:
        for train_input, test_input in [
                ("train/sonar-train.txt", "test/sonar-test.txt"),
                ("train/splice-train.txt", "test/splice-test.txt")]:

            train_features, train_labels = read_data(train_input)
            test_features, test_labels = read_data(test_input)
            dataset = train_input.split('/')[-1]

            # k = -1 means use raw data without any reduction method
            for k_dimension in [-1, 10, 20, 30]:
                predict(reduction_method, train_features, train_labels, test_features, test_labels,
                        k_dimension, dataset=dataset)

if __name__ == "__main__":
    main()