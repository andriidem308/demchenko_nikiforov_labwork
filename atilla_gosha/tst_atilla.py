import matplotlib.pyplot as plt
import numpy as np
import utils_atilla as utils


data = np.load('data.npz')
x, y = data['x'], data['y']
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
t = np.arange(-1, 1, 0.01).reshape((-1, 1))
t = np.concatenate([np.ones((t.shape[0], 1)), t], axis=1)

x_small, y_small = x[:15], y[:15]


class LR_valid_L1:
    def __init__(self, num_features=1, optimizer=utils.GD(0.05), alpha=0):
        self.W = np.zeros((num_features, 1)) + 5
        self.optimizer = optimizer
        self.alpha = alpha

    def predict(self, X):
        y = X @ self.W
        return y

    def one_step_opt(self, X, y_true):
        grad_reg = self.alpha * np.sign(self.W)
        grad_reg[0, 0] = 0
        reg_loss = np.sum(self.alpha * np.abs(self.W))
        grads = - X.T @ (y_true - X @ self.W) / X.shape[0] + grad_reg
        print(grads.shape)
        exit()
        self.W = self.optimizer.apply_grads(self.W, grads)
        loss = (y_true - self.predict(X)).T @ (y_true - self.predict(X)) / X.shape[0]
        return loss, grads, reg_loss

    def fit(self, X, y_true, X_valid=None, y_valid=None, grad_tol=0.0001, n_iters=10000):
        grad_norm = np.inf
        n_iter = 0
        losses = []
        valid_losses = []
        reg_losses = []
        while (grad_norm > grad_tol) and (n_iter < n_iters):
            loss, grads, reg_loss = self.one_step_opt(X, y_true)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            valid_loss = (y_valid - self.predict(X_valid)).T @ (y_valid - self.predict(X_valid)) / X_valid.shape[0]

            losses.append(loss[0][0])
            valid_losses.append(valid_loss[0][0])
            reg_losses.append(reg_loss)
        return np.array(losses), np.array(valid_losses), np.array(reg_losses)

    def fit_closed_form(self, X, y_true, X_valid=None, y_valid=None):
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y_true
        loss = (y_true - self.predict(X)).T @ (y_true - self.predict(X)) / X.shape[0]
        return loss