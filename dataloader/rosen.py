from scipy.optimize import rosen
import numpy as np


class RosenData:
    def __init__(self):
        pass

    def dataset(self, n_train, n_val, n_test, n_pool, n_dim):
        X_train = np.random.random((n_train, n_dim))
        y_train = self._rosen(X_train)

        X_val = np.random.random((n_val, n_dim))
        y_val = self._rosen(X_val)

        X_test = np.random.random((n_test, n_dim))
        y_test = self._rosen(X_test)

        X_pool = np.random.random((n_pool, n_dim))
        y_pool = self._rosen(X_pool)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_pool, y_pool

    @staticmethod
    def _rosen(x):
        return rosen(x.T)[:, None]
