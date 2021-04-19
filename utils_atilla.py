import numpy as np

class GD:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def apply_grads(self, W, grad_W):
        W = W - self.lr * grad_W
        return W

class LR:
    def __init__(self, num_features=1, optimizer=GD(0.1)):
        self.W = np.zeros((num_features, 1))
        self.optimizer = optimizer
    
    def predict(self, X):
        y = X @ self.W
        return y
    
    def one_step_opt(self, X, y_true):
        grads = - X.T @ (y_true - X @ self.W) / X.shape[0]
        self.W = self.optimizer.apply_grads(self.W, grads)
        loss = (y_true - self.predict(X)).T @ (y_true - self.predict(X))
        return loss, grads
    
    def fit(self, X, y_true, grad_tol=0.0001, n_iters=1000):
        grad_norm = np.inf
        n_iter = 0
        losses = []
        while (grad_norm > grad_tol) and (n_iter < n_iters):
            loss, grads = self.one_step_opt(X, y_true)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            losses.append(loss[0][0])
        return losses
    
    def fit_closed_form(self, X, y_true):
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y_true
        loss = (y_true - self.predict(X)).T @ (y_true - self.predict(X)) / X.shape[0]
        return loss