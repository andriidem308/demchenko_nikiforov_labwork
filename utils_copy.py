import numpy as np


class LR:
    def __init__(self, w_init, b_init):
        self.w = w_init
        self.b = b_init
        self.w_h = []
        self.b_h = []
        self.e_h = []
        # self.algo = algo

    def sigmoid(self, x, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return 1. / (1. + np.exp(-(w * x + b)))

    def error(self, X, Y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        err = 0
        for x, y in zip(X, Y):
            err += 0.5 * (self.sigmoid(x, w, b) - y) ** 2
        return err

    def grad_w(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred)

    def fit(self, X, Y,
            epochs=32, eta=0.01, gamma=0.9, mini_batch_size=32, eps=1e-8,
            beta=0.9, beta1=0.9, beta2=0.9):
        self.w_h = []
        self.b_h = []
        self.e_h = []
        self.X = X
        self.Y = Y

        v_w, v_b = 0, 0
        for i in range(epochs):
            dw, db = 0, 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            v_w = gamma * v_w + eta * dw
            v_b = gamma * v_b + eta * db
            self.w = self.w - v_w
            self.b = self.b - v_b
            self.append_log()

    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
        self.e_h.append(self.error(self.X, self.Y))