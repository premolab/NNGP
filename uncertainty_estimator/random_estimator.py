import numpy as np


class RandomEstimator:
    def estimate(self, X_pool, *args, **kwargs):
        return np.zeros_like(X_pool)
