import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import tqdm
import time
from scipy.optimize import rosen, rosen_der
import scipy
import matplotlib.pyplot as plt


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def get_errors(x_, y_):
    return [np.sqrt(mse(x_, y_)), np.mean(np.abs(x_ - y_)), np.max(np.abs(x_ - y_))]


def simple_cov(_x, _y):
    return np.mean((_x - np.mean(_x)) * (_y - np.mean(_y)), axis=1)


if __name__ == '__main__':

    df = pd.read_csv('../../slice_localization_data.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(['patientId'], axis=1, inplace=True)
    target_label = 'reference'
    targets = df[target_label].values
    df.drop([target_label], axis=1, inplace=True)
    train_num = 10000
    thres = 42000
    thres2 = len(df)

    X_train = df[:train_num].values
    y_train = targets[:train_num][:, None]
    X_pool = df[train_num:thres].values
    y_pool = targets[train_num:thres][:, None]
    X_test = df[thres:thres2].values
    y_test = targets[thres:thres2][:, None]
    print('shapes:', X_train.shape, y_train.shape)

    tf.reset_default_graph()
    # layers
    ndim = X_train.shape[1]
    layers = [256, 128, 128]

    x = tf.placeholder(tf.float32, [None, ndim])
    y_ = tf.placeholder(tf.float32, [None, 1])

    learning_rate_ = tf.placeholder(tf.float32)
    forces_coeff_ = tf.placeholder(tf.float32)
    keep_probability_ = tf.placeholder(tf.float32, name='keep_probability')
    l2_reg_ = tf.placeholder(tf.float32, name='l2reg')

    # weights
    W1 = tf.Variable(tf.truncated_normal([ndim, layers[0]], stddev=(2 / ndim) ** .5))
    b1 = tf.Variable(tf.truncated_normal([layers[0]], stddev=.1))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    h_drop1 = tf.nn.dropout(h1, keep_probability_, noise_shape=[1, layers[0]])

    Ws = [W1]
    bs = [b1]
    hs = [h_drop1]
    for cnt_layer in range(1, len(layers)):
        Ws.append(tf.Variable(tf.truncated_normal([layers[cnt_layer - 1], layers[cnt_layer]],
                                                  stddev=(2 / layers[cnt_layer - 1]) ** .5)))
        bs.append(tf.Variable(tf.truncated_normal([layers[cnt_layer]], stddev=.1)))
        hs.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hs[-1], Ws[-1]) + bs[-1]),
                                keep_probability_,
                                noise_shape=[1, layers[cnt_layer]]))

    Ws.append(tf.Variable(tf.truncated_normal([layers[-1], 1], stddev=.1)))
    bs.append(tf.Variable(tf.truncated_normal([1], stddev=.1)))

    # funcs
    y = tf.matmul(hs[-1], Ws[-1]) + bs[-1]

    l2_regularizer = sum(tf.nn.l2_loss(Wxxx) for Wxxx in Ws)

    mse_e = tf.losses.mean_squared_error(predictions=y, labels=y_)
    loss = mse_e + l2_reg_ * l2_regularizer

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 8e-4
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # init sess
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    batch_size = 500
    init_epochs = 50000
    # init_epochs = 500
    keep_prob = .95
    l2_reg = 5e-5

    uptrain_epochs = 200
    sample_each_step = 200
    T = 25
    al_iterations = 20

    gpnn_max_train = 1000
    diag_eps = .001
    points_to_integrate = 1000

    saver.restore(sess, "../../init_model.ckpt")
    print("Init model restored")
    epoch = 0
    data = []
    X_train_current = X_train.copy()
    y_train_current = y_train.copy()
    X_pool_current = X_pool.copy()
    y_pool_current = y_pool.copy()

    print('=' * 40)
    print('Integrated MSE Gain-Max Variance')
    print('=' * 40)

    for al_iters in range(al_iterations):
        print('Starting GPNN iteration #', al_iters)
        random_train_inds = np.random.permutation(range(len(X_train_current)))[:gpnn_max_train]
        train_pool_samples = np.concatenate([X_train_current[random_train_inds], X_pool_current])
        stds = np.zeros((train_pool_samples.shape[0], T), dtype=float)
        for cnt_ in range(T):
            stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: train_pool_samples,
                                                            keep_probability_: .5}))

        K_train_cov = np.cov(stds[:gpnn_max_train, :], ddof=0)
        K_train_cov_inv = np.linalg.inv(K_train_cov + np.eye(gpnn_max_train) * diag_eps)

        train_samples = X_train_current[random_train_inds]

        minimums = train_samples.min(axis=0)
        maximums = train_samples.max(axis=0)

        vs = np.random.uniform(minimums, maximums,
                               size=(points_to_integrate, train_samples.shape[1]))

        y_vs = np.zeros((vs.shape[0], T), dtype=float)

        for cnt_ in range(T):
            y_vs[:, cnt_] = np.ravel(sess.run(y,
                                              feed_dict={x: vs, keep_probability_: .5}))

        sigmas = []
        for cnt_ in range(len(vs)):
            vs_sample = y_vs[cnt_, :]
            Q = simple_cov(stds[:gpnn_max_train], vs_sample)[:, None]
            KK = np.var(vs_sample)
            sigma = KK - np.dot(np.dot(Q.T, K_train_cov_inv), Q)[0][0]
            sigmas.append(np.sqrt(sigma))

        diffs_integral = []
        new_K_cov = np.zeros((gpnn_max_train + 1, gpnn_max_train + 1))
        new_K_cov[:gpnn_max_train, :gpnn_max_train] = K_train_cov

        for x_cnt_ in tqdm.tqdm(range(len(X_pool_current))):
            pool_sample = stds[(gpnn_max_train + x_cnt_), :]
            Q = simple_cov(stds[:gpnn_max_train, :], pool_sample)[:, None]
            Q = Q.ravel()
            new_K_cov[-1, :-1] = Q
            new_K_cov[:-1, -1] = Q
            new_K_cov[-1, -1] = np.var(pool_sample)
            new_K_cov_inv = np.linalg.inv(new_K_cov + np.eye(gpnn_max_train + 1) * diag_eps)

            indices = list(range(gpnn_max_train)) + [gpnn_max_train + x_cnt_]

            extended_sigmas = []
            for cnt_ in range(len(y_vs)):
                vs_sample = y_vs[cnt_, :]
                Q = simple_cov(stds[indices], vs_sample)[:, None]
                KK = np.var(vs_sample)
                sigma = KK - np.dot(np.dot(Q.T, new_K_cov_inv), Q)[0][0]
                extended_sigmas.append(np.sqrt(sigma))

            current_diff = np.array(sigmas) - np.array(extended_sigmas)
            diffs_integral.append(current_diff.sum())
            print("STOPPED")
            print("TODO: select best points and add to train set")
            break
        break
