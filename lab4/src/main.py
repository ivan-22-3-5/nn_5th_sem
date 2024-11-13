from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def plot_data(classification_data, classification_labels,
              regression_data, regression_targets):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(classification_data[:, 0], classification_data[:, 1], c=classification_labels, cmap='viridis',
                edgecolor='k')
    plt.title('Classification Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    plt.scatter(regression_data[:, 0], regression_data[:, 1], c=regression_targets[:, 0], cmap='plasma', edgecolor='k')
    plt.title('Regression Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()


def main():
    n = 100 + 2 * 18
    classification_data, classification_labels = datasets.make_classification(n_samples=n, n_features=2, n_redundant=0,
                                                                              n_classes=3, n_clusters_per_class=1
                                                                              )
    regression_data, regression_targets = datasets.make_regression(n_samples=n, n_features=4, n_targets=2)

    plot_data(classification_data, classification_labels,
              regression_data, regression_targets)

    x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
        classification_data, classification_labels, test_size=0.2, random_state=42
    )

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        regression_data, regression_targets, test_size=0.2, random_state=42
    )


if __name__ == '__main__':
    main()
