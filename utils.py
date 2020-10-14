import numpy as np


m_init = np.zeros((5, 1))  # change 5 to FEATURES and move to main.ipynb
alpha = np.full((5, 1), 0.1)  # change 5 to FEATURES and move to main.ipynb
beta = np.full((5, 1), 0.9)  # change 5 to FEATURES and move to main.ipynb


def normalize(features):
    # try to implement this method without for loop
    # find method in NumPy Library
    new_features = np.asarray(features).T
    for feature in new_features:
        min_f = np.min(feature)
        max_f = np.max(feature)

        for i in range(len(feature)):
            feature[i] = (feature[i] - min_f) / (max_f - min_f)

    return new_features.T


class Momentum:
    def __init__(self, num_features=5):
        self.m = np.zeros((num_features, 1))

    def regularization(self, weight, grad):
        self.m = beta * self.m + (1 - beta) * grad
        weight = weight - (alpha * self.m)
        return weight


class Model:
    def __init__(self, batch_size=32, num_features=5):
        self.weights = np.random.rand(num_features, 1)
        self.batch_size = batch_size
        self.optimizer = Momentum(num_features)

    def predict(self, x):
        y = x @ self.weights
        return y

    def optimize_once(self, x, y_true):
        # fix this gradient !!!
        grads = x.T @ (y_true - x @ self.weights) / x.shape[0]
        grads = np.reshape(grads.T[0], (5, 1))

        self.weights = self.optimizer.regularization(self.weights, grads)
        loss = (y_true - self.predict(x)).T @ (y_true - self.predict(x))

        return loss, grads

    def fit(self, x, y_true, grad_tol=0.01, epochs=100):
        x_train = x[:self.batch_size]
        y_train = y_true[:self.batch_size]
        grad_norm = np.inf
        n_iter = 0
        losses = []
        while (grad_norm > grad_tol) and (n_iter < epochs):
            loss, grads = self.optimize_once(x_train, y_train)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            losses.append(loss)

        # return losses
