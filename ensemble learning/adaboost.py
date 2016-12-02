import numpy as np
from naive_bayes import NaiveBayes


class AdaBoost(object):
    def __init__(self, iteration=10):
        self.weights = None
        self.classifier = []

        self.iteration = iteration

    def _init_weights(self, size):
        self.weights = np.mat(np.full((size, 1), 1 / float(size)))

    def _normal_weights(self):
        total = np.sum(self.weights)
        self.weights = self.weights / total

    def _get_sample(self, size):
        weights = [self.weights[j, 0] for j in range(self.weights.shape[0])]
        return [np.random.choice(size, p=weights) for _ in range(size)]

    def train(self, data, label, types):
        size, dimension = data.shape
        self._init_weights(size)

        for i in range(self.iteration):
            sample_index = self._get_sample(size)
            sample_data = data[sample_index]
            sample_label = label[sample_index]

            classifier = NaiveBayes(sample_data, sample_label, data, types)
            classifier.train()
            class_result = classifier.test(data)

            false_index = np.where(class_result != label)[0]
            error = np.sum(self.weights[false_index])

            if error > 0.5 or error < 0.00001:
                break

            alpha = 0.5 * np.log((1.0 - error) / error)
            self.weights = np.multiply(self.weights, np.exp(alpha * np.multiply(class_result, label)))
            self._normal_weights()
            self.classifier.append((classifier, alpha))

    def test(self, data):
        result = np.zeros((data.shape[0], 1))
        for classifier, alpha in self.classifier:
            class_result = classifier.test(data)
            result += class_result * alpha

        return np.sign(result)






