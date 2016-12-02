import numpy as np

from utils import random_list, load_matrix_from_txt, draw
from regression import lr_iteration, ridge_iteration, compute_loss


lamda = 0.01
gamma = 0.0001
max_iteration = 5
step = 0.01


def error(beta, data, label):
    n = data.shape[0]

    true_positive = 0
    for i in range(n):
        if beta * data[i].T * label[i] > 0:
            true_positive += 1

    return 1 - float(true_positive) / float(n)


def sgd(train_data, train_label, test_data, test_label, method, max_iteration=5, step=0.01):
    n, d = train_data.shape

    # beta = np.random.random((1, d))
    beta = np.zeros((1, d))
    beta[:, :] = 0.01
    train_errors = []
    test_errors = []
    losses = []
    iteration_count = 0
    sample = 0.0

    for i in range(max_iteration):
        index_list = random_list(n)

        for i in range(n):
            index = index_list.pop()

            if method == 'logistic':
                beta = lr_iteration(beta, train_data[index], train_label[index], lamda, gamma)
            elif method == 'ridge':
                beta = ridge_iteration(beta, train_data[index], train_label[index], lamda, gamma)

            if np.math.floor(max_iteration * n * sample) == iteration_count:
                train_error = error(beta, train_data, train_label)
                test_error = error(beta, test_data, test_label)
                loss = compute_loss(beta, train_data, train_label, method, lamda)
                train_errors.append(train_error)
                test_errors.append(test_error)
                losses.append(loss)

                print "t: {}, loss:{}, error: {}".format(sample, loss, train_error)
                sample += float(step)

            iteration_count += 1
    return beta, train_errors, test_errors, losses


def main():
    for train_file, test_file in [
        # ("data/dataset1-a9a-training.txt", "data/dataset1-a9a-testing.txt"),
                                  ("data/covtype-training.txt", "data/covtype-testing.txt")
                                  ]:
        train_data, train_label = load_matrix_from_txt(train_file)
        test_data, test_label = load_matrix_from_txt(test_file)

        for method in ['logistic']:
            _, train_errors, test_errors, losses = sgd(train_data, train_label, test_data, test_label, method, max_iteration=max_iteration, step=step)
            draw(train_errors, test_errors, losses, train_file, max_iteration=max_iteration, step=step, lamda=lamda, gamma=gamma)


if __name__ == "__main__":
    main()
