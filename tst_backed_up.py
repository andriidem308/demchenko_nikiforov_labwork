import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Momentum:
    def __init__(self, num_features=5, alpha=0.7, beta=0.9):
        self.m = np.zeros((num_features, 1))
        self.alpha = alpha
        self.beta = beta

    def optimize_weights(self, weights, grads):
        self.m = self.beta * self.m + (1 - self.beta) * grads
        new_weights = weights - (self.alpha * self.m)

        return new_weights


class Model:
    def __init__(self, num_features=5, alpha=0.03):
        self.weights = np.zeros((num_features, 1)) + 5
        self.optimizer = Momentum()
        self.num_features = num_features
        self.alpha = alpha

    def predict(self, x):
        return self.sigmoid(x @ self.weights)

    @staticmethod
    def sigmoid(z):
        y = 1 / (1 + np.exp(-z))
        return y

    def optimize_once(self, x, y):
        loss_reg = np.sum(self.alpha * np.abs(self.weights))
        grad_reg = np.sum(self.alpha * np.abs(self.weights))

        ones = np.ones(y.shape)
        hypothesis = self.predict(x)
        grads = (-y @ np.log(hypothesis) - (ones - y) @ np.log(ones - hypothesis)) / x.shape[0]  # + grad_reg

        # self.weights = self.optimizer.optimize_weights(self.weights, grads)
        loss_train = (y - hypothesis).T @ (y - hypothesis) / x.shape[0]

        return loss_train, grads, loss_reg

    def fit(self, x, y, batch_size=32, grad_tol=0.001, epochs=50):
        self.x_train = x[:batch_size]
        self.y_train = y[:batch_size]
        self.x_test = x[batch_size:]
        self.y_test = y[batch_size:]

        grad_norm = np.inf
        n_iter = 0
        losses_train, losses_test, losses_reg  = [], [], []

        while (grad_norm > grad_tol) and (n_iter < epochs):
            loss_train, grads, loss_reg = self.optimize_once(self.x_train, self.y_train)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            loss_test = (self.y_test - self.predict(self.x_test)).T @ (self.y_test - self.predict(self.x_test)) / self.x_test.shape[0]

            losses_train.append(loss_train[0][0])
            losses_test.append(loss_test)
            losses_reg.append(loss_reg)

        return np.array(losses_train), np.array(losses_test), np.array(losses_reg)

    def predict_proba(self, x):
        print(self.predict(x))


def normalize(features):
    new_features = np.asarray(features).T
    min_f = np.amin(new_features, axis=1)
    max_f = np.amax(new_features, axis=1)

    new_features = (new_features.T - min_f) / (max_f - min_f)

    return new_features


if __name__ == "__main__":
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    # print(dataset.head())

    features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
    X_init = df[features]
    Y_init = df["DEATH_EVENT"]

    X = normalize(X_init)
    Y = np.asarray(Y_init).copy()

    model = Model()
    model.fit(X, Y)
    model.predict_proba(X[100:120])
    print(Y[100:120])
