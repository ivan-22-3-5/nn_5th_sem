from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.perceptron import Perceptron

generate_data = partial(make_blobs, centers=2, cluster_std=0.7, random_state=0)


def plot_data(points, point_classes):
    plt.scatter(points[:, 0], points[:, 1], c=point_classes, cmap='bwr', edgecolor='k')
    plt.title('Points')
    plt.show()


def test_perceptron(x_test, y_test, perceptron):
    correct = np.sum([round(perceptron.predict(x)) for x in x_test] == y_test)
    return correct / len(y_test)


def main():
    x, y = generate_data(118)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

    plot_data(x, y)

    perceptron = Perceptron(input_size=2)

    print(f'Initial weights: {perceptron.weights}')
    print(f'Accuracy before training: {test_perceptron(x_test, y_test, perceptron) * 100:.2f}%')
    perceptron.train((x_train, y_train))
    print(f'Weights after training: {perceptron.weights}')
    print(f'Accuracy after training: {test_perceptron(x_test, y_test, perceptron) * 100:.2f}%')


if __name__ == '__main__':
    main()
