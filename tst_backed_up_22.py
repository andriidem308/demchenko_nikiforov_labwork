import pandas as pd
import numpy as np

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

class Momentum:
    def __init__(self, num_features=1, alpha=0.2, beta=0.9):
        self.m = np.zeros((num_features, 1))
        self.alpha = alpha
        self.beta = beta

    def optimize_weights(self, weights, grads):
        self.m = self.beta * self.m + self.alpha * grads
        new_weights = weights - self.m

        return new_weights

class LogR:
    """
    Constructor

    params:
    num_features - number of features
    optimizer - optimizer
    alpha - reguralization coefficient
    """

    def __init__(self, num_features=1, alpha=0.001):
        self.W = np.zeros((num_features, 1))
        self.M, self.V, self.t = 0, 0, 1
        self.optimizer = Momentum(num_features)
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
        min_f = np.amin(X, axis=0)
        max_f = np.amax(X, axis=0)
        result = (X - min_f) / (max_f - min_f)
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
        grad_reg = self.alpha * np.sign(self.W)
        grad_reg[0, 0] = 0
        reg_loss = np.sum(self.alpha * np.abs(self.W)) ###
        grads = - X.T @ (y_true - X @ self.W) / X.shape[0] + grad_reg ###
        self.W = self.optimizer.optimize_weights(self.W, grads)
        loss = np.sum((y_true - self.predict(X)).T @ (y_true - self.predict(X))) / X.shape[0]

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

    def create_fold(self, X, Y, folds, k):
        features = np.c_[X, Y]
        r = int(features.shape[0] / folds)
        if k == 0:
            x1 = np.array([])
        else:
            x1 = features[0:k * r, :]
        if k == r - 1:
            x3 = np.array([])
        else:
            x3 = features[k * r + r:, :]
        valid = features[k * r:(k * r + r), :]
        if k == 0:
            train = x3
        elif k == r - 1:
            train = x1
        else:
            train = np.concatenate([x1, x3], axis=0)
        x_valid, y_valid = valid[:, :-1], valid[:, -1].reshape(-1, 1)
        x_train, y_train = train[:, :-1], train[:, -1].reshape(-1, 1)
        return x_train, y_train, x_valid, y_valid

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

    def fold_fit(self, x_train, y_train, x_valid, y_valid, n_iters=10000, epsil=0.0001):
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
        y_t = self.predict(x_valid)
        estimation = np.sum(np.where(y_valid == y_true, 1, 0)) / y_valid.shape[0]
        return losses, valid_losses, estimation, y_true, y_t


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
titles = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
          'ejection_fraction', 'high_blood_pressure', 'platelets',
          'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

# import seaborn as sns
# sns.pairplot(df, vars=titles, hue="DEATH_EVENT", dropna=True, diag_kind="hist")
#
# ##############
# exit()


feature_names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'serum_sodium', 'sex', 'smoking']
size = len(feature_names)

X_init = df[feature_names]
Y_init = df['DEATH_EVENT']

lr = LogR()

npx = np.array(X_init).reshape((299, size))

x = lr.normalize_std(npx)
y = np.array(Y_init)


features = np.c_[x, y]
np.random.shuffle(features)

BATCH_SIZE = 64
test_ex = 10

x_train = features[:BATCH_SIZE, :size].reshape((BATCH_SIZE,size))
y_train = features[:BATCH_SIZE, size:].reshape((BATCH_SIZE,1))

x_test = features[BATCH_SIZE:BATCH_SIZE+test_ex, :size].reshape((test_ex,size))
y_test = features[BATCH_SIZE:BATCH_SIZE+test_ex, size:].reshape((test_ex,1))

# print(np.c_[x_train, y_train])
lr = LogR(size)
losses, valid_losses, estimation, y_true, y_t = lr.fold_fit(x_train, y_train, x_test, y_test)

print(np.c_[y_test, y_t, y_true])
# print(y_test.shape)
# print(y_t.shape)
# print(y_true.shape)
print(losses)
print(valid_losses)
print(estimation)
#
