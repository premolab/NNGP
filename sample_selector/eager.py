import numpy as np


class EagerSampleSelector:
    def update_sets(self, X_train, y_train, X_pool, y_pool, ue_values, sample_size):
        inds = self.sample(ue_values, sample_size)
        X_train = np.concatenate([X_train, X_pool[inds]])
        y_train = np.concatenate([y_train, y_pool[inds]])
        X_pool = np.delete(X_pool, inds, axis=0)
        y_pool = np.delete(y_pool, inds, axis=0)

        return X_train, y_train, X_pool, y_pool

    @staticmethod
    def sample(values, sample_size):
        return np.argsort(values)[::-1][:sample_size]
