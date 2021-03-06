#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class LogR:
    """
    Constructor

    params:
    num_features - number of features
    optimizer - optimizer
    alpha - reguralization coefficient
    """

    def __init__(self, num_features=1, optimizer=ADAM(), alpha=0):
        self.W = np.zeros((num_features, 1))
        self.M, self.V, self.t = 0, 0, 1
        self.optimizer = optimizer
        self.eps = 0.000001
        self.alpha = alpha
        # self.eps = 0

    """
    Predict

    returns probabilty of belonging to the first class

    params:
    X - features
    """

    def predict(self, X):
        y = 1 / (1 + np.exp(- (X @ self.W)))
        return y

    """
    Predict_proba

    returns array with 0 and 1: if element belongs to the first class - 1, else 0

    params:
    X - features
    """

    def predict_proba(self, X):
        y = self.predict(X)
        return np.where(y >= 0.5, 1, 0)

    """
    conv

    returns a small number - absolute value of difference bitween means of two last elements

    params:
    arr - array of arrays
    """

    def conv(self, arr):
        res = np.inf
        if len(arr) > 2:
            res = np.abs(np.mean(arr) - np.mean(arr[:-1]))
        return res

    """
    normalize_std

    returns a normalized matrix

    params:
    X - matrix
    """

    def normalize_std(self, X):
        X = X[:, 1:]
        result = X - np.mean(X, axis=0)
        result = result / np.std(X, axis=0)
        result = np.c_[np.ones((result.shape[0], 1)), result]
        return result

    """
    one_step_opt

    returns a loss, grads and reg loss
    applies optimazer and does one step of optimization

    params:
    X - features
    y_true - dependent variables
    """

    def one_step_opt(self, X, y_true):
        grad_reg = 2 * self.alpha * self.W
        grad_reg[0, 0] = 0
        reg_loss = np.sum(self.alpha * ((self.W) ** 2))
        grads = - X.T @ (y_true - self.predict(X)) / X.shape[0] + grad_reg
        self.W = self.optimizer.apply_grads(self.W, grads)
        loss = np.sum(-y_true * np.log(self.predict(X) + self.eps) -
                      (1 - y_true) * np.log(1 - self.predict(X) + self.eps)) / X.shape[0]
        return loss, grads, reg_loss

    """
    batch_split

    returns randomly splitted on equal-size blocks features with dependent variables

    params:
    X - features
    Y - dependent variables
    size - block size
    """

    def batch_split(self, X, Y, size):
        features = np.c_[X, Y]
        np.random.shuffle(features)
        features = np.array(np.split(features, range(size, features.shape[0], size)))
        return features

    """
    create_fold

    creates a train and valid data for k-fold cross validation
    applies to the model number k

    returns two pairs
    first pair - x_train (features of train dataset) and y_train (dependent variables of train dataset)
    second pair - x_valid (features of validation dataset) and y_valid (dependent variables of validation dataset)

    params:
    X - features
    Y - dependent variables
    folds - number of folds
    k - current model
    """



    """
    fit_batches

    trains model 

    returns losses and reg losses after training

    params:
    X - features
    y_true - dependent variables
    size - block size
    n_iters - limiting number of iteration
    epsil - limiting value of conv
    """

    def fit_batches(self, X, y_true, size=32, n_iters=20000, epsil=0.000005):
        n_iter = 0
        losses = []
        reg_losses = []
        while (n_iter < n_iters) and (self.conv(losses) > epsil):
            n_iter += 1
            current_losses = []
            current_reg_losses = []
            mini_batches = self.batch_split(X, y_true, size)
            for batch in mini_batches:
                x_small, y_small = batch[:, :-1], batch[:, -1].reshape(-1, 1)
                loss, grads, reg_loss = self.one_step_opt(x_small, y_small)
                current_losses.append(loss)
                current_reg_losses.append(reg_loss)
            losses.append(np.mean(current_losses))
            reg_losses.append(np.mean(current_reg_losses))
        return np.array(losses), np.array(reg_losses)

    """
    fold_fit

    returns losses and regularization losses after training on train dataset;
    estimation of correct predictions from validation dataset

    params:
    x_train - features of train dataset
    y_train - dependent variables of train dataset
    x_valid - features of validation dataset
    y_valid - dependent variables of validation dataset
    n_iters - limiting number of iteration
    epsil - limiting value of conv
    """

    def fold_fit(self, x_train, y_train, x_valid, y_valid, n_iters=20000, epsil=0.000001):
        losses = []
        valid_losses = []
        n_iter = 0
        while (n_iter < n_iters) and (self.conv(losses) > epsil):
            n_iter += 1
            t_loss, grads, reg_loss = self.one_step_opt(x_train, y_train)
            losses.append(t_loss)
            v_loss = np.sum(-y_valid * np.log(self.predict(x_valid) + self.eps) -
                            (1 - y_valid) * np.log(1 - self.predict(x_valid) + self.eps)) / x_valid.shape[0]
            valid_losses.append(v_loss)
        y_true = self.predict_proba(x_valid)
        estimation = np.sum(np.where(y_valid == y_true, 1, 0)) / y_valid.shape[0]
        return losses, valid_losses, estimation

    def fit(self, X, y_true, grad_tol=0.0001, n_iters=20000):
        grad_norm = np.inf
        n_iter = 0
        losses = []
        while (grad_norm > grad_tol) and (n_iter < n_iters):
            loss, grads = self.one_step_opt(X, y_true)
            grad_norm = np.linalg.norm(grads)
            n_iter += 1
            losses.append(loss)
        return losses
