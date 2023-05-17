import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm


# scroll to the bottom to start coding your solution
def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')
    plt.show()


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, X):
        return sigmoid(np.dot(X, self.weights) + self.biases)

    def backprop(self, X, y, alpha):
        error = (mse_prime(self.forward(X), y) * sigmoid_prime(np.dot(X, self.weights) + self.biases))

        delta_W = np.dot(X.T, error) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        self.weights -= alpha * delta_W
        self.biases -= alpha * delta_b


class TwoLayerNeural:
    def __init__(self, n_features, n_classes, hidden_size=64):
        self.weights = [xavier(n_features, hidden_size), xavier(hidden_size, n_classes)]
        self.biases = [xavier(1, hidden_size), xavier(1, n_classes)]

    def forward(self, X):
        values = X
        for i in range(2):
            values = sigmoid(np.dot(values, self.weights[i]) + self.biases[i])
        return values

    def backprop(self, X, y, alpha):
        n = X.shape[0]
        biases = np.ones((1, n))

        y_pred = self.forward(X)

        loss_grad_1 = 2 * alpha / n * ((y_pred - y) * y_pred * (1 - y_pred))

        f1_out = sigmoid(np.dot(X, self.weights[0]) + self.biases[0])

        loss_grad_0 = np.dot(loss_grad_1, self.weights[1].T) * f1_out * (1 - f1_out)

        self.weights[0] -= np.dot(X.T, loss_grad_0)
        self.weights[1] -= np.dot(f1_out.T, loss_grad_1)

        self.biases[0] -= np.dot(biases, loss_grad_0)
        self.biases[1] -= np.dot(biases, loss_grad_1)


def scale(X_train, X_test):
    X_max = np.max(X_train)
    return X_train / X_max, X_test / X_max


def xavier(n_in, n_out):
    return np.random.uniform(
        -np.sqrt(6) / np.sqrt(n_in + n_out),
        np.sqrt(6) / np.sqrt(n_in + n_out),
        (n_in, n_out)
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(x, y):
    return np.mean((x - y) ** 2)


def mse_prime(x, y):
    return 2 * (x - y)


def accuracy(estimator, X, y):
    y_pred = np.argmax(estimator.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true), mse(y_pred, y_true)


def train(estimator, X, y, alpha, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)


if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train, X_test = scale(X_train, X_test)

    model = TwoLayerNeural(X_train.shape[1], y_train.shape[1])

    r2 = []
    losses = [accuracy(model, X_test, y_test)[1]]
    for _ in tqdm(range(50)):
        train(model, X_train, y_train, 0.5)
        acc_res = accuracy(model, X_test, y_test)
        r2.append(acc_res[0])
        losses.append(acc_res[1])
    print(r2)
    plot(losses, r2)
