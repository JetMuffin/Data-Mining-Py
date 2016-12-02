import numpy as np

from utils import load_matrix_from_txt, k_fold_split, compute_accuracy
from adaboost import AdaBoost


def main():
    for file in [
        'data/breast-cancer-assignment5.txt',
        'data/german-assignment5.txt'
    ]:
        data, labels, types = load_matrix_from_txt(file)
        splices = k_fold_split(10, data, labels)
        accuracies = []

        for i in range(10):
            train_indexes = splices[i][0]
            test_indexes = splices[i][1]

            train_data = np.copy(data[train_indexes])
            train_label = np.copy(labels[train_indexes])
            test_data = np.copy(data[test_indexes])
            test_label = np.copy(labels[test_indexes])

            boost = AdaBoost()
            boost.train(train_data, train_label, types)
            class_result = boost.test(test_data)

            accuracy = compute_accuracy(class_result, test_label)
            accuracies.append(accuracy)
            print 'accuracy: %f' % accuracy

        print('file: {}, mean: {}, std: {}'.format(file, np.mean(accuracies), np.std(accuracies)))


if __name__ == '__main__':
    main()