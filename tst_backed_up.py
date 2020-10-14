import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
FEATURES = 5
m_init = np.zeros((FEATURES, 1))
alpha = np.full((FEATURES, 1), 0.5)
beta = np.full((FEATURES, 1), 0.5)


def normalize(features):
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
    def __init__(self, batch_size=BATCH_SIZE, num_features=5, alpha=0.004):
        self.weights = np.zeros((num_features, 1))
        self.optimizer = Momentum(5)
        # print(self.weights)
        self.alpha = alpha
        self.batch_size = batch_size

    def predict(self, x):
        y = x @ self.weights
        return y

    def optimize_once(self, x, y_true):
        # fix this gradient
        grad_reg = self.alpha * np.sign(self.weights)
        grad_reg[0, 0] = 0
        # reg_loss = np.sum(self.alpha * np.abs(self.weights))
        # grads = x.T @ (y_true - x @ self.weights) / x.shape[0]
        grads = - x.T @ (y_true - x @ self.weights) / x.shape[0] + grad_reg
        print(grads.shape)
        grads = np.reshape(grads.T[0], (5, 1))
        self.weights = self.optimizer.regularization(self.weights, grads)
        loss = (y_true - self.predict(x)).T @ (y_true - self.predict(x)) / x.shape[0]
        # print(loss)

        return loss, grads

    def fit(self, x, y_true, x_valid=None, y_valid=None, grad_tol=0.0001, epochs=1000):
        x_train = x[:self.batch_size]
        y_train = y_true[:self.batch_size]
        grad_norm = np.inf
        n_iter = 0
        losses = []
        valid_losses = []
        while (grad_norm > grad_tol) and (n_iter < epochs):
            loss, grads = self.optimize_once(x_train, y_train)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            valid_loss = ((y_valid) - self.predict(x_valid)).T @ (y_valid - self.predict(x_valid)) / x_valid.shape[0]

            losses.append(loss[0][0])

        return losses


if __name__ == "__main__":
    dataset = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    # print(dataset.head())

    alphas = np.arange(0.009, 0.01, 0.0001)
    features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
    X_init = dataset[features]
    Y_init = dataset["DEATH_EVENT"]

    X = normalize(X_init)
    Y = np.asarray(Y_init).copy()
    x_valid = X[32:]
    y_valid = Y[32:]

    model = Model(BATCH_SIZE, alpha=0.9)
    model.fit(X, Y, x_valid, y_valid)
    y_pred = model.predict(X[32:100])

    print("y pred shape:", y_pred.shape)
    # print("y pred: ", y_pred)
    print("y pred: ")
    for i in range(32, 100):
        print(y_pred[32 - i], Y[i])



