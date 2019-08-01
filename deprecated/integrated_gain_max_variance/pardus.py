from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import time
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import sys
import os
import time
import pickle
from numba import jit

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

@jit
def simple_cov(_x, _y):
    return np.mean((_x-np.mean(_x))*(_y-np.mean(_y)), axis = 1)

def get_mcdues(X):
    stds = np.zeros((X.shape[0], params['T']), dtype = float)
    for cnt_ in range(params['T']):
        stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: X, 
                                                        keep_probability_: .5}))
    return np.std(stds, axis = 1)

def get_stds(X):
    stds = np.zeros((X.shape[0], T), dtype = float)
    for cnt_ in range(T):
        stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: X, 
                                                        keep_probability_: .5}))
    return stds

@jit
def compute_block_inv(A_inv, B, C, D):
    H = D - C.dot(A_inv).dot(B)
    H_inv = 1./ H
    a00 = A_inv + H_inv * A_inv.dot(B).dot(C).dot(A_inv)
    a01 = -A_inv.dot(B) * H_inv
    a10 = -H_inv * C.dot(A_inv)
    a11 = H_inv
    
    return np.block([[a00, a01.reshape(-1, 1)],
                    [a10.reshape((1, -1)), a11[0]]])

paramx = int(sys.argv[-1])
#paramx = 4
dataset = paramx % 6


if dataset == 0:
    df = pd.read_csv('../data/network.data', header = None)
    df.columns = ['X'+str(x) for x in range(df.shape[1])]
    df.drop(['X0','X10','X15'], axis = 1, inplace = True) # zero st.d.
    target_label = 'X23'
elif dataset == 1:
    df = pd.read_csv('../data/slice_localization_data/slice_localization_data.csv')
    df.drop(['patientId'], axis = 1, inplace = True)
    target_label = 'reference'
elif dataset == 2:
    df = pd.read_csv('../data/YearPredictionMSD.txt/YearPredictionMSD.txt', header = None)
    df.columns = ['X' + str(x) for x in range(df.shape[1])]
    target_label = 'X0'
elif dataset == 3:
    df = pd.read_csv('../data/sgemm_product_dataset/sgemm_product.csv')
    df['run'] = np.median(df.values[:,-4:], axis = 1)
    df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis = 1, inplace = True)
    target_label = 'run'
elif dataset == 4:
    df = pd.read_csv('../data/OnlineNewsPopularity/OnlineNewsPopularity.csv')
    df.drop(['url'], axis = 1, inplace = True)
    target_label = ' shares'
elif dataset == 5:
    df = pd.read_csv('../data/rosen2KD.csv')
    target_label = 'ans'

with open(str(paramx) + '/params.pickle', 'rb') as f:
    params = pickle.load(f)

df = df.loc[params['df_index']].reset_index(drop = True)
targets = df[target_label].values    
df.drop([target_label], axis = 1, inplace = True)

# workaround for my stupidity
# params['N_val'] = params['N_test']
# params['N_pool'] = 8*params['N_train']

X_train = df[:params['N_train']].values
y_train = targets[:params['N_train']][:, None]
X_test = df[params['N_train']:(params['N_train'] + params['N_test'])].values
y_test = targets[params['N_train']:(params['N_train'] + params['N_test'])][:, None]
X_pool = df[-params['N_pool']:].values
y_pool = targets[-params['N_pool']:][:, None]
X_val = df[-(params['N_val']+params['N_pool']):-params['N_pool']].values
y_val = targets[-(params['N_val']+params['N_pool']):-params['N_pool']][:, None]

print('shapes:', X_train.shape, y_train.shape)
print('shapes:', X_test.shape, y_train.shape)
print('shapes:', X_pool.shape, y_pool.shape)
print('shapes:', X_val.shape, y_val.shape)

# NN architecture

tf.reset_default_graph()

# placeholders
x = tf.placeholder(tf.float32, [None, params['ndim']])
y_ = tf.placeholder(tf.float32, [None, 1])

learning_rate_ = tf.placeholder(tf.float32)
forces_coeff_ = tf.placeholder(tf.float32)
keep_probability_ = tf.placeholder(tf.float32, name='keep_probability')
l2_reg_ = tf.placeholder(tf.float32, name='l2reg')

W1 = tf.Variable(tf.truncated_normal([params['ndim'], params['layers'][0]], 
    stddev=(2/params['ndim'])**.5))
b1 = tf.Variable(tf.truncated_normal([params['layers'][0]],  stddev=.1))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h_drop1 = tf.nn.dropout(h1, keep_probability_, noise_shape = [1,params['layers'][0]])

Ws = [W1]; bs = [b1]; hs = [h_drop1]
for cnt_layer in range(1, len(params['layers'])):
    Ws.append(tf.Variable(tf.truncated_normal([params['layers'][cnt_layer - 1], 
                                            params['layers'][cnt_layer]], 
                                            stddev=(2/params['layers'][cnt_layer - 1])**.5)))
    bs.append(tf.Variable(tf.truncated_normal([params['layers'][cnt_layer]],  stddev=.1)))
    hs.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hs[-1], Ws[-1]) + bs[-1]), keep_probability_,
                            noise_shape = [1,params['layers'][cnt_layer]]))

Ws.append(tf.Variable(tf.truncated_normal([params['layers'][-1], 1], stddev=.1)))
bs.append(tf.Variable(tf.truncated_normal([1],  stddev=.1)))

y = tf.matmul(hs[-1], Ws[-1]) + bs[-1]

l2_regularizer = sum(tf.nn.l2_loss(Wxxx) for Wxxx in Ws) 

mse_e = tf.losses.mean_squared_error(predictions = y, labels = y_)
loss = mse_e + l2_reg_*l2_regularizer

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params['start_learning_rate']
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        params['learning_rate_schedule_epochs'],
                                        params['learning_rate_decay'], staircase=True)

lr_fun = lambda: learning_rate
min_lr = lambda: tf.constant(1e-5)
actual_lr = tf.case([(tf.less(learning_rate, tf.constant(1e-5)), min_lr)], default=lr_fun)

train_step = tf.train.AdamOptimizer(learning_rate=actual_lr).minimize(loss, global_step=global_step)

try:
    sess.close()
except:
    pass

acc_mode = 'np'
w8s_folder = "/home/Maxim.Panov/nngp/w8s/" if acc_mode == 'p' else "/home/Evgenii.Tsymbalov/activelearning/gpnn/w8s/"

NUM_THREADS = int(os.environ['PBS_NP'])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS))
sess.run(init)
epoch = 0
data = []

saver.restore(sess, w8s_folder + "init_" + params['fname_identifier'] + ".ckpt")

X_train_current = X_train.copy()
y_train_current = y_train.copy()
#X_pool_current = X_pool[:1500,:].copy()
#y_pool_current = y_pool[:1500,:].copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

@jit
def dodot(Q, W):
    return np.dot(np.dot(Q.T, W), Q)[0][0]

@jit
def fxjit(x_cnt_):
    pool_sample = stds[(gpnn_max_train + x_cnt_), :]
    #t = time.time()
    Q = simple_cov(stds[:gpnn_max_train, :], pool_sample)[:, None]
    Q = Q.ravel()
    #print('Q in', np.round(time.time()-t, 2)); t=time.time()
    new_K_cov[-1, :-1] = Q
    new_K_cov[:-1, -1] = Q
    new_K_cov[-1, -1] = np.var(pool_sample)
    #print('Q insert in', np.round(time.time()-t, 2)); t=time.time()
    new_K_cov_inv = compute_block_inv(K_train_cov_inv,
                                      Q.reshape((-1, 1)),
                                      Q.reshape((1, -1)), 
                                      np.var(pool_sample) + params['diag_eps'])
    #print('new_K_cov_inv in', np.round(time.time()-t, 2)); t=time.time()
    indices = list(range(gpnn_max_train)) + [gpnn_max_train + x_cnt_]
    si = stds[indices]
    ### count sigma(v | X + x_from_pool) with extended 
    ### cov matrix for each v in vs
    extended_sigmas = np.zeros((len(y_vs),))
    for cnt_ in range(len(y_vs)):
        vs_sample = y_vs[cnt_, :]
        Q = simple_cov(si, vs_sample)[:, None]
        KK = np.var(vs_sample)
        #sigma = KK + params['diag_eps'] - np.dot(np.dot(Q.T, new_K_cov_inv), Q)[0][0]
        sigma = KK - dodot(Q, new_K_cov_inv) + diag_eps
        extended_sigmas[cnt_] = np.sqrt(sigma)
    #print('integrate in', np.round(time.time()-t, 2)); t=time.time()
    current_diff = np.array(sigmas) - np.array(extended_sigmas)
    #print('misc in', np.round(time.time()-t, 2)); t=time.time()
    return current_diff.sum()


points_to_integrate = 1000
print('='*40)
print('imse-based ALGO')
print('='*40)
gpnn_max_train = params['gpnn_max_train']
T = params['T']
diag_eps = params['diag_eps']
for al_iters in range(params['al_steps']):
#for al_iters in range(2):
    t = time.time()
    print('Starting iteration #', al_iters)
    perm = np.random.permutation(range(len(X_train_current)))
    random_train_inds = perm[:gpnn_max_train]
    random_train_samples = X_train_current[random_train_inds]
    train_and_pool_samples = np.concatenate([random_train_samples, X_pool_current])    
    stds = get_stds(train_and_pool_samples)
    K_train_cov = np.cov(stds[:gpnn_max_train, :], ddof = 0)
    K_train_cov_inv = np.linalg.inv(K_train_cov + diag_eps * np.eye(gpnn_max_train))

    vs = X_pool_current[-points_to_integrate:,:]
    y_vs = get_stds(vs)
    print('Work on sigmas at #', al_iters)
    ### sigma(v | X) for each v in vs
    sigmas = np.zeros((len(y_vs),))
    for cnt_ in range(len(vs)):
        vs_sample = y_vs[cnt_, :]
        Q = simple_cov(stds[:gpnn_max_train], vs_sample)[:, None]
        KK = np.var(vs_sample)
        sigma = KK - np.dot(np.dot(Q.T, K_train_cov_inv), Q)[0][0]
        sigmas[cnt_] = np.sqrt(sigma)

    diffs_integral = np.zeros(X_pool_current.shape[0])
    new_K_cov = np.zeros((gpnn_max_train + 1, gpnn_max_train + 1))
    new_K_cov[:gpnn_max_train, :gpnn_max_train] = K_train_cov

    print('Work on ext sigmas at #', al_iters)
    for x_cnt_ in range(len(X_pool_current)-points_to_integrate):
        diffs_integral[x_cnt_] = fxjit(x_cnt_)
        if x_cnt_ % 10 == 0:
            print(x_cnt_, end = '|')

    inds = np.argsort(diffs_integral)[::-1][:params['sample_each_step']]
    print('Added to training set, new sizes:', X_train_current.shape, y_train_current.shape)
    # 3) add them to the training set
    X_train_current = np.concatenate([X_train_current, X_pool_current[inds, :]])
    y_train_current = np.concatenate([y_train_current, y_pool_current[inds, :]])
    print('Added to training set, new sizes:', X_train_current.shape, y_train_current.shape)
    # 4) remove them from the pool
    X_pool_current = np.delete(X_pool_current, inds, axis = 0)
    y_pool_current = np.delete(y_pool_current, inds, axis = 0)
    print('Deleted from pool set, new sizes:', X_pool_current.shape, y_pool_current.shape)
    # 5) uptrain the NN
    prev_test_error = 1e+10
    sample_selection_time = time.time() - t
    t_big = time.time()
    t = time.time()
    for cnt in range(params['uptrain_epochs']):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, params['batch_size']):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: params['keep_prob'], 
                                            l2_reg_: params['l2_reg']})
        # checking errors
        if (cnt+1) % params['early_stopping_check_step'] == 0:
            print(np.round(time.time() - t, 2), end='s')
            t = time.time()
            preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
            preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

            train_err =  get_errors(preds_train, y_train_current)
            test_err =  get_errors(preds_test, y_test)
            print(' &', np.round(time.time() - t, 2), 's')
            print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
            #data.append([al_iters] + train_err + test_err)
            # checking early stopping conditions
            if (test_err[0] > prev_test_error*(1 + params['early_stopping_window'])) and (cnt > params['mandatory_uptrain_epochs']):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= params['max_warnings']:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, w8s_folder + "imse_" + params['fname_identifier']  + ".ckpt")
                print("GPNN imse model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, w8s_folder + "imse_" + params['fname_identifier']  + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = params['data_columns']).copy()
    datadf.to_csv('csvs/data_imse_' + params['fname_identifier']  + '.csv', index = False)
