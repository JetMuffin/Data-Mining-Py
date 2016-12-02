import numpy as np


class NaiveBayes(object):
    def __init__(self, data, labels, all_data, types):
        self.labels = np.unique(np.array(labels))
        self.train_data = data
        self.train_labels = labels
        self.all_data = all_data
        self.types = types

        self.train_model = dict()

    def train(self):
        size, dimension = self.train_data.shape

        for label in self.labels:
            idx = np.argwhere(self.train_labels == label)[:, 0]
            class_data = self.train_data[idx]
            class_data_number = class_data.shape[0]

            # compute prior probability `P(C = c)` of each class
            prior_probability = (1.0 * class_data_number + 1) / (size + len(self.labels))
            self.train_model[(label, 'priori_probability')] = prior_probability

            # compute conditional probability `P(xj = aj | C = c)` of each feature
            for i in range(dimension):
                # laplacian correction
                Ni = len(np.unique(np.array(self.all_data[:, i])))

                # numeric features
                if self.types[i] == 0:
                    data_numeric = class_data[:, i]
                    mean = np.mean(data_numeric)
                    var = np.var(data_numeric)
                    self.train_model[(label, i)] = (mean, var)
                # discrete features
                else:
                    vals = np.unique(np.array(self.all_data[:, i]))
                    for val in vals:
                        count_in_class = len(np.where(class_data[:, i] == val)[0])
                        self.train_model[(label, i, val)] = (count_in_class + 1) / float(class_data_number + Ni)

    def test(self, data):
        size, dimension = data.shape

        test_probability_all_labels = np.mat(np.full((size, len(self.labels)), 1.0, dtype=np.float))
        for index, label in enumerate(self.labels):
            test_probability_all_labels[:, index] = self._condition_probability(data, label)

        predict = np.argmax(test_probability_all_labels, axis=1)
        for i in range(size):
            predict[i] = self.labels[predict[i]]

        return predict

    def _condition_probability(self, data, label):
        size, dimension = data.shape
        test_probability = np.mat(np.full((size, 1), 1.0, dtype=np.float))
        for j in range(dimension):
            if self.types[j] == 0:
                test_probability = np.multiply(test_probability, np.mat(np.mat(self._gaussian(label, j, data[:, j])).T))
            else:
                for i in range(size):
                    test_probability[i] *= self.train_model[(label, j, data[i, j])]

        return test_probability

    def _gaussian(self, label, index, val):
        mean, var = self.train_model[(label, index)]

        # var will be zero after several times of sampling
        if var == 0:
            p = np.zeros(val.shape)
            for i in range(p.shape[0]):
                if val[i] == mean:
                    p[i] = 1
                else:
                    p[i] = 0
            return p

        return 1.0 / (np.sqrt(2 * np.pi * var)) * np.exp(- np.square(val - mean) / (2 * var))



