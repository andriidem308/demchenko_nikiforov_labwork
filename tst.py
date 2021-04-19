import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Momentum:
    def __init__(self, num_features=1, alpha=0.1):
        self.m = np.zeros((num_features, 1))
        self.alpha = alpha

    def optimize_weights(self, weights, grads):
        self.m = (1 - self.alpha) * self.m + self.alpha * grads
        new_weights = weights - self.alpha * self.m

        return new_weights

class LogR:
    def __init__(self, num_features=1, alpha=0.1):
        self.W = np.zeros((num_features, 1))
        self.M, self.V, self.t = 0, 0, 1
        self.optimizer = Momentum(num_features, alpha)
        self.eps = 0.0001
        self.alpha = alpha
        # self.eps = 0

    def predict(self, X):
        y = 1 / (1 + np.exp(- (X @ self.W)))
        return y

    def predict_proba(self, X):
        y = self.predict(X)
        return np.where(y >= 0.5, 1, 0)

    def conv(self, arr):
        res = np.inf
        if len(arr) > 2:
            res = np.abs(np.mean(arr) - np.mean(arr[:-1]))
        return res

    def normalize_std(self, X):
        min_f = np.amin(X, axis=0)
        max_f = np.amax(X, axis=0)
        result = (X - min_f) / (max_f - min_f)
        return result

    def one_step_opt(self, X, y_true):
        grad_reg = self.alpha * np.sign(self.W)
        grad_reg[0, 0] = 0
        reg_loss = np.sum(self.alpha * np.abs(self.W))
        grads = - X.T @ (y_true - X @ self.W) / X.shape[0] + grad_reg
        self.W = self.optimizer.optimize_weights(self.W, grads)
        loss = np.sum((y_true - self.predict(X)).T @ (y_true - self.predict(X))) / X.shape[0]

        return loss, grads, reg_loss

    def batch_split(self, X, Y, size):
        features = np.c_[X, Y]
        np.random.shuffle(features)
        features = np.array(np.split(features, range(size, features.shape[0], size)))
        return features

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
    def fit(self, x_train, y_train, x_valid=None, y_valid=None, n_iters=1000, epsil=0.00001):
        losses = []
        n_iter = 0
        while (n_iter < n_iters) and (self.conv(losses) > epsil):
            n_iter += 1
            t_loss, grads = self.one_step_opt(x_train, y_train)

            losses.append(t_loss)

        return losses

    def fold_fit(self, x_train, y_train, x_valid, y_valid, n_iters=10000, epsil=0.0001):
        losses = []
        valid_losses = []
        n_iter = 0
        while (n_iter < n_iters) and (self.conv(losses) > epsil):
            n_iter += 1
            t_loss, grads, reg_losses = self.one_step_opt(x_train, y_train)
            v_loss = np.sum(-y_valid * np.log(self.predict(x_valid) + self.eps) -
                            (1 - y_valid) * np.log(1 - self.predict(x_valid) + self.eps)) / x_valid.shape[0]

            losses.append(t_loss)
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


FEATURE_NAMES = ['age', 'anaemia', 'high_blood_pressure', 'serum_sodium', 'sex', 'smoking']
# FEATURE_NAMES = ['age', 'anaemia', 'high_blood_pressure']
# FEATURE_NAMES = titles[:]
size = len(FEATURE_NAMES)

x_df = np.array(df[FEATURE_NAMES]).reshape((299, size))
y_df = np.asarray(df['DEATH_EVENT'])

xy_shuffled = np.c_[x_df, y_df]
np.random.shuffle(xy_shuffled)

x_shuffled, y_shuffled = xy_shuffled[:, :size], xy_shuffled[:, size:]

lr = LogR(size, alpha=0.1)

x = lr.normalize_std(x_shuffled)
y = y_shuffled

BSIZE = 200
N_TESTS = 20

x_train, y_train, = x[:BSIZE], y[:BSIZE].reshape(BSIZE, 1)
x_test, y_test = x[BSIZE:BSIZE + N_TESTS], y[BSIZE:BSIZE + N_TESTS].reshape(N_TESTS, 1)

losses, valid_losses, estimation, y_true, y_t = \
    lr.fold_fit(x_train, y_train, x_test, y_test, n_iters=100000, epsil=0.0001)
print(np.c_[y_test, y_t, y_true])
# print(lr.predict(x_test))
print(losses)
print(valid_losses)
print(estimation)


# losses,reg_losses = lr.fit_batches(x_train, y_train)
# y_true = lr.predict(x_test)
# print(losses)
# print(y_true)




plt.plot(losses)
plt.show()