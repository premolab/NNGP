import numpy as np


class MCDUE:
    def __init__(self, net):
        self.net = net

    def estimate(self, session, X_pool, nn_runs=25, probability=.5, use_inner=False):
        mcd_realizations = np.zeros((X_pool.shape[0], nn_runs))
        probability_inner = probability if use_inner else 1.
        
        for nn_run in range(nn_runs):
            prediction = self.net.predict(
                session, data=X_pool, probability=probability, probabitily_inner=probability_inner
            )
            mcd_realizations[:, nn_run] = np.ravel(prediction)
            
        return np.ravel(np.std(mcd_realizations, axis=1))
