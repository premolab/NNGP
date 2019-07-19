import tensorflow as tf
import numpy as np


class NNGP:
    def __init__(
        self, net, random_subsampling=-1, nn_runs=25, diag_eps=1e-6,
        inference_batch_size=1000, probability=.5, use_inner=False
    ):
        self.net = net
        self.random_subsampling = random_subsampling
        self.nn_runs = nn_runs
        self.diag_eps = diag_eps
        self.inference_batch_size = inference_batch_size
        self.probability = probability
        self.use_inner = use_inner

        self.KK_ = tf.placeholder(tf.float32)
        self.Q_ = tf.placeholder(tf.float32, [None, None])
        self.K_train_cov_inv_ = tf.placeholder(tf.float32, [None, None])
        define_one = tf.matmul(tf.transpose(self.Q_), self.K_train_cov_inv_)  # what is this?
        define_two = tf.matmul(define_one, self.Q_)                           # and what is that?
        self.gpue_tf = tf.linalg.tensor_diag_part(self.KK_ - define_two)


    def estimate(self, session, X_train, y_train, X_pool):
        # data preparation
        train_pool_samples, train_len = self._pool(X_train, X_pool)

        # nn inference
        mcd_realizations = np.zeros((train_pool_samples.shape[0], self.nn_runs))
        for nn_run in range(self.nn_runs):
            prediction = self._net_predict(session, train_pool_samples)
            mcd_realizations[:, nn_run] = np.ravel(prediction)

        # covariance matrix with regularization
        cov_matrix_train = np.cov(mcd_realizations[:train_len, :], ddof=0)
        cov_matrix_inv = np.linalg.inv(cov_matrix_train + np.eye(train_len)*self.diag_eps)

        gp_ue = np.zeros((len(X_pool), ))

        cnt = 0 # for the case of inference_batch_size > len(X_pool)
        for cnt in range(len(X_pool) // self.inference_batch_size):
            left_ind = train_len + cnt*self.inference_batch_size
            right_ind = train_len + cnt*self.inference_batch_size+self.inference_batch_size
            pool_samples = mcd_realizations[left_ind:right_ind,:]
            Qs = self.simple_covs(mcd_realizations[:train_len,:], pool_samples).T
            KKs = np.var(pool_samples, axis=1)
            ws = session.run(self.gpue_tf, {
                             self.Q_: Qs,
                             self.K_train_cov_inv_: cov_matrix_inv,
                             self.KK_: KKs})
            gp_ue_currents = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
            gp_ue[(left_ind - train_len):(right_ind - train_len)] = gp_ue_currents
        right_ind = train_len + cnt*self.inference_batch_size+self.inference_batch_size
        pool_samples = mcd_realizations[right_ind:,:]
        Qs = self.simple_covs(mcd_realizations[:train_len,:], pool_samples).T
        KKs = np.var(pool_samples, axis=1)
        ws = session.run(self.gpue_tf, {
                             self.Q_: Qs,
                             self.K_train_cov_inv_: cov_matrix_inv,
                             self.KK_: KKs})
        gp_ue_currents = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
        gp_ue[right_ind:] = gp_ue_currents
        return np.ravel(gp_ue)

    def _pool(self, X_train, X_pool):
        train_len = len(X_train)
        if self.random_subsampling > 0:
            train_len = min(self.random_subsampling, train_len)
            random_train_inds = np.random.permutation(range(train_len))[:self.random_subsampling]
            train_pool_samples = np.concatenate([X_train[random_train_inds], X_pool])
        else:
            train_pool_samples = np.concatenate([X_train, X_pool])

        return train_pool_samples, train_len

    def _net_predict(self, session, train_pool_samples):
        probabitily_inner = self.probability if self.use_inner else 1.
        return self.net.predict(
            session, data=train_pool_samples, probability=self.probability,
            probabitily_inner=probabitily_inner
        )
    
    @staticmethod
    def simple_covs(a, b):
        ac = a - a.mean(axis=-1, keepdims=True)
        bc = (b - b.mean(axis=-1, keepdims=True)) / b.shape[-1]
        return np.dot(ac, bc.T).T

