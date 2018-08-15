
# coding: utf-8

# In[4]:


from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import time
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import sys
import time

dataset = int(sys.argv[-1])

# In[2]:


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


# In[6]:


def get_errors(x_, y_):
    return [np.sqrt(mse(x_, y_)), np.mean(np.abs(x_ - y_)), np.max(np.abs(x_ - y_))]

def simple_cov(_x, _y):
    return np.mean((_x-np.mean(_x))*(_y-np.mean(_y)), axis = 1)

def get_mcdues(X):
    stds = np.zeros((X.shape[0], T), dtype = float)
    for cnt_ in range(T):
        stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: X, 
                                                        keep_probability_: .5}))
    return np.std(stds, axis = 1)


# # Getting data

# In[ ]:


#df = pd.read_csv('../AL/data/slice_localization_data/slice_localization_data.csv')
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

df = df.sample(frac=1).reset_index(drop=True) # TODO: save these indices
targets = df[target_label].values    
df.drop([target_label], axis = 1, inplace = True)

# TODO: parametrize data set
N_total = len(df)
N_train, N_test, N_val, N_pool = (int(tmp*N_total) for tmp in (.1, .05, .05, .8))


X_train = df[:N_train].values
y_train = targets[:N_train][:, None]
X_test = df[N_train:(N_train + N_test)].values
y_test = targets[N_train:(N_train + N_test)][:, None]
X_pool = df[-N_pool:].values
y_pool = targets[-N_pool:][:, None]
X_val = df[-(N_val+N_pool):-N_pool].values
y_val = targets[-(N_val+N_pool):-N_pool][:, None]

print('shapes:', X_train.shape, y_train.shape)
print('shapes:', X_test.shape, y_train.shape)
print('shapes:', X_pool.shape, y_pool.shape)
print('shapes:', X_val.shape, y_val.shape)


# # NN architecture parameters

# In[ ]:


ndim = X_train.shape[1]
# layers = [64,32]
layers = [256,128,128]

learning_rate_decay = .97
start_learning_rate = 8e-4
learning_rate_schedule_epochs = 50000

# TODO: add weight stddev and stuff


# In[5]:


tf.reset_default_graph()

# placeholders
x = tf.placeholder(tf.float32, [None, ndim])
y_ = tf.placeholder(tf.float32, [None, 1])

learning_rate_ = tf.placeholder(tf.float32)
forces_coeff_ = tf.placeholder(tf.float32)
keep_probability_ = tf.placeholder(tf.float32, name='keep_probability')
l2_reg_ = tf.placeholder(tf.float32, name='l2reg')

# weights
W1 = tf.Variable(tf.truncated_normal([ndim, layers[0]], stddev=(2/ndim)**.5))
b1 = tf.Variable(tf.truncated_normal([layers[0]],  stddev=.1))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h_drop1 = tf.nn.dropout(h1, keep_probability_, noise_shape = [1,layers[0]])

Ws = [W1]; bs = [b1]; hs = [h_drop1]
for cnt_layer in range(1, len(layers)):
    Ws.append(tf.Variable(tf.truncated_normal([layers[cnt_layer - 1], layers[cnt_layer]], 
                                              stddev=(2/layers[cnt_layer - 1])**.5)))
    bs.append(tf.Variable(tf.truncated_normal([layers[cnt_layer]],  stddev=.1)))
    hs.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hs[-1], Ws[-1]) + bs[-1]), keep_probability_,
                            noise_shape = [1,layers[cnt_layer]]))

Ws.append(tf.Variable(tf.truncated_normal([layers[-1], 1], stddev=.1)))
bs.append(tf.Variable(tf.truncated_normal([1],  stddev=.1)))

# funcs
y = tf.matmul(hs[-1], Ws[-1]) + bs[-1]

l2_regularizer = sum(tf.nn.l2_loss(Wxxx) for Wxxx in Ws) 

mse_e = tf.losses.mean_squared_error(predictions = y, labels = y_)
loss = mse_e + l2_reg_*l2_regularizer

#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = start_learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           learning_rate_schedule_epochs, learning_rate_decay, staircase=True)

lr_fun = lambda: learning_rate
min_lr = lambda: tf.constant(1e-5)
actual_lr = tf.case([(tf.less(learning_rate, tf.constant(1e-5)), min_lr)], default=lr_fun)

train_step = tf.train.AdamOptimizer(learning_rate=actual_lr).minimize(loss, global_step=global_step)


# In[ ]:


try:
    sess.close()
except:
    pass

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
epoch = 0
data = []

fname_identifier = '_' + str(dataset) + '_' + hex(int(np.random.random()*1e+12))[2:].upper()
# fname_identifier = '28D3AA0A1D'


# In[7]:


batch_size = 200
init_epochs = 1000000
keep_prob = .9
l2_reg = 1e-4

al_steps = 20
uptrain_epochs = 1000000
mandatory_uptrain_epochs = 10000
sample_each_step = 200
T = 25

early_stopping_window = .03
max_warnings = 3
early_stopping_check_step = 100

gpnn_max_train = 1000
diag_eps = .001

data_columns = ['al_step', 'sample_selection_time', 'uptraining_time', 'train_rmse',
    'train_mae', 'train_maxae', 'test_rmse', 'test_mae', 'test_maxae']


# In[ ]:


X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()


# # Initial_training

# In[ ]:


lr, gs = sess.run([learning_rate, global_step])
print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
prev_test_error = 1e+10
t = time.time()
for cnt in range(init_epochs):
    epoch += 1
    # training itself
    
    for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
        X_batch, y_batch = batch
        sess.run(train_step, feed_dict={x: X_batch, 
                                        y_: y_batch, 
                                        keep_probability_: keep_prob, 
                                        l2_reg_: l2_reg})
    # checking errors
    if (cnt+1) % early_stopping_check_step == 0:
        print(np.round(time.time() - t, 2), end='s')
        t = time.time()
        preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
        preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})
        
        train_err =  get_errors(preds_train, y_train_current)
        test_err =  get_errors(preds_test, y_test)
        print(' &', np.round(time.time() - t, 2), 's')
        print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
        data.append([epoch] + train_err + test_err)
        # checking early stopping conditions
        if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
            warnings += 1
            print('*'*warnings, end = '||')
            if warnings >= max_warnings:
                print('$$$')
                break
        else:
            warnings = 0
            prev_test_error = min(test_err[0], prev_test_error)
        t = time.time()
lr, gs = sess.run([learning_rate, global_step])
print('learning rate: {:.4E}, global step: {}'.format(lr, gs))


# In[ ]:


save_path = saver.save(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model saved in path: %s" % save_path)


# In[ ]:


datadf = pd.DataFrame(data, columns = ['epoch', 'train_rmse',
    'train_mae', 'train_maxae', 'test_rmse', 'test_mae', 'test_maxae']).copy()
datadf.to_csv('csvs/data_init' + fname_identifier + '.csv', index = False)


# # MCDUE itself

# In[ ]:


saver.restore(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model restored")

data = []
X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

print('='*40)
print('MCDUE-based ALGO')
print('='*40)
for al_iters in range(al_steps):
    t = time.time()
    # 1) get MCDUEs
    print('Starting AL iteration #', al_iters)
    mcdues = get_mcdues(X_pool_current)
    print('AL iteration #', al_iters, ': got MCDUEs')
    # 2) pick n_pick samples with top mcdues
    inds = np.argsort(mcdues)[::-1][:sample_each_step]
    print(sample_each_step, 'samples picked')
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
    for cnt in range(uptrain_epochs):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: keep_prob, 
                                            l2_reg_: l2_reg})
        # checking errors
        if (cnt+1) % early_stopping_check_step == 0:
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
            if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= max_warnings:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, "/tmp/mcdue_" + fname_identifier + ".ckpt")
                print("MCDUE model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/tmp/mcdue_" + fname_identifier + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf.to_csv('csvs/data_mcdue' + fname_identifier + '.csv', index = False)

# save_path = saver.save(sess, "/tmp/mcdue_" + fname_identifier + ".ckpt")
# print("MCDUE model saved in path: %s" % save_path)


# # RANDOM

# In[ ]:


saver.restore(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model restored")
epoch = 0
data = []
X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

print('='*40)
print('RANDOM-based ALGO')
print('='*40)
for al_iters in range(al_steps):
    t = time.time()
    print('Starting RANDOM iteration #', al_iters)
    # 2) pick n_pick samples with top mcdues
    inds = np.random.permutation(len(X_pool_current))[:sample_each_step]
    print(sample_each_step, 'samples picked')
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
    for cnt in range(uptrain_epochs):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: keep_prob, 
                                            l2_reg_: l2_reg})
        # checking errors
        if (cnt+1) % early_stopping_check_step == 0:
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
            if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= max_warnings:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, "/tmp/random_" + fname_identifier + ".ckpt")
                print("RANDOM model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/tmp/random_" + fname_identifier + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf.to_csv('csvs/data_random' + fname_identifier + '.csv', index = False)

#save_path = saver.save(sess, "/tmp/random_" + fname_identifier + ".ckpt")
#print("RANDOM model saved in path: %s" % save_path)


# # KMMCDUE

# In[ ]:


saver.restore(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model restored")

data = []
X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

print('='*40)
print('KMMCDUE-based ALGO')
print('='*40)
for al_iters in range(al_steps):
    t = time.time()
    # 1) get MCDUEs
    print('Starting AL iteration #', al_iters)
    mcdues = get_mcdues(X_pool_current)
    print('AL iteration #', al_iters, ': got MCDUEs')
    # 2) pick n_pick samples with top mcdues
    km_model = KMeans(n_clusters = sample_each_step, verbose=2)
    inds = np.argsort(mcdues)[::-1][::-1]
    km_model.fit(X_pool_current[inds[:int(0.1*X_train_current.shape[0])]]) # KMeans on top 10%
    print('Fitted KMeans with', sample_each_step, 'clusters')
    inds, _ = pairwise_distances_argmin_min(km_model.cluster_centers_, X_pool_current)
    print(sample_each_step, 'samples picked')
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
    for cnt in range(uptrain_epochs):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: keep_prob, 
                                            l2_reg_: l2_reg})
        # checking errors
        if (cnt+1) % early_stopping_check_step == 0:
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
            if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= max_warnings:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, "/tmp/kmmcdue_" + fname_identifier + ".ckpt")
                print("KMMCDUE model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/tmp/kmmcdue_" + fname_identifier + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf.to_csv('csvs/data_kmmcdue' + fname_identifier + '.csv', index = False)
#save_path = saver.save(sess, "/tmp/kmmcdue_" + fname_identifier + ".ckpt")
#print("KMMCDUE model saved in path: %s" % save_path)


# # GPNN

# In[ ]:


saver.restore(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model restored")
epoch = 0
data = []
X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

print('='*40)
print('GPNN-based ALGO')
print('='*40)
for al_iters in range(al_steps):
    t = time.time()
    # 1) get MCDUEs
    print('Starting GPNN iteration #', al_iters)
    random_train_inds = np.random.permutation(range(len(X_train_current)))[:gpnn_max_train]
    train_pool_samples = np.concatenate([X_train_current[random_train_inds], X_pool_current])
    stds = np.zeros((train_pool_samples.shape[0], T), dtype = float)
    for cnt_ in range(T):
        stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: train_pool_samples, 
                                                        keep_probability_: .5}))
    print('Got MCDUEs')
    K_train_cov = np.cov(stds[:gpnn_max_train, :], ddof = 0)
    K_train_cov_inv = np.linalg.inv(K_train_cov + np.eye(gpnn_max_train)*diag_eps)
    print('Got and inverted COV matrix')
    gp_ue = []
#     t = time.time()
    for cnt in range(len(X_pool_current)):
#         if (1+cnt) % 100 == 0:
#             print(int(cnt*100./len(X_pool_current)), '(', np.round(time.time() - t, 1), 's)', end = '.')
#             t = time.time()
        pool_sample = stds[(gpnn_max_train+cnt),:]
        Q = simple_cov(stds[:gpnn_max_train,:], pool_sample)[:, None]
        KK = np.var(pool_sample)
        gp_ue_current = np.sqrt(KK - np.dot(np.dot(Q.T, K_train_cov_inv), Q)[0][0])
        gp_ue.append(gp_ue_current)
    print('Calculated the GPUEs')
    gp_ue = np.array(gp_ue).T
    inds = np.argsort(gp_ue)[::-1][:sample_each_step]
    print(sample_each_step, 'samples picked')
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
    for cnt in range(uptrain_epochs):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: keep_prob, 
                                            l2_reg_: l2_reg})
        # checking errors
        # checking errors
        if (cnt+1) % early_stopping_check_step == 0:
            print(np.round(time.time() - t, 2), end='s')
            t = time.time()
            preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
            preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

            train_err =  get_errors(preds_train, y_train_current)
            test_err =  get_errors(preds_test, y_test)
            print(' &', np.round(time.time() - t, 2), 's')
            print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
            # 0 + train_err + test_err)
            # checking early stopping conditions
            if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= max_warnings:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, "/tmp/gpnn_" + fname_identifier + ".ckpt")
                print("GPNN model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/tmp/gpnn_" + fname_identifier + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf.to_csv('csvs/data_gpnn' + fname_identifier + '.csv', index = False)

#save_path = saver.save(sess, "/tmp/gpnn_" + fname_identifier + ".ckpt")
#print("GPNN model saved in path: %s" % save_path)


# # 2-step GPNN

# In[ ]:


saver.restore(sess, "/tmp/init_" + fname_identifier + ".ckpt")
print("Init model restored")
epoch = 0
data = []
X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()

print('='*40)
print('2-step GPNN-based ALGO')
print('='*40)
for al_iters in range(5):
    t = time.time()
    # 1) get MCDUEs
    print('Starting 2-step GPNN iteration #', al_iters)
    XX_train_current = X_train_current.copy()
    XX_pool_current = X_pool_current.copy()

    inds = []
    indices = pd.Series(range(len(XX_pool_current)))
    for gp_cnt in range(10):
        random_train_inds = np.random.permutation(range(len(XX_train_current)))[:gpnn_max_train]
        train_pool_samples = np.concatenate([XX_train_current[random_train_inds], XX_pool_current])
        stds = np.zeros((train_pool_samples.shape[0], T), dtype = float)
        for cnt_ in range(T):
            stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: train_pool_samples, 
                                                            keep_probability_: .5}))
        K_train_cov = np.cov(stds[:gpnn_max_train, :], ddof = 0)
        K_train_cov_inv = np.linalg.inv(K_train_cov + np.eye(gpnn_max_train)*diag_eps)
        gp_ue = []
        for cnt in range(len(XX_pool_current)):
            pool_sample = stds[(gpnn_max_train+cnt),:]
            Q = simple_cov(stds[:gpnn_max_train,:], pool_sample)[:, None]
            KK = np.var(pool_sample)
            gp_ue.append(np.sqrt(KK - np.dot(np.dot(Q.T, K_train_cov_inv), Q)[0][0]))
        gp_ue = np.array(gp_ue).T
        inds_temporal = np.argsort(gp_ue)[::-1][:(sample_each_step // 10)]
        inds += list(indices[inds_temporal])
        print(gp_cnt, end = '|')
        XX_train_current = np.concatenate([XX_train_current, XX_pool_current[inds_temporal, :]])
        XX_pool_current = np.delete(XX_pool_current, inds_temporal, axis = 0)

        for i in inds_temporal:
            del indices[i]
        indices = indices.reset_index(drop = 1)
    print('Finished 2-step GP sampling')
    print(len(inds), 'samples picked; check~', len(set(inds)))
    
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
    for cnt in range(uptrain_epochs):
        epoch += 1
        # training itself

        for batch in iterate_minibatches(X_train_current, y_train_current, batch_size):
            X_batch, y_batch = batch
            sess.run(train_step, feed_dict={x: X_batch, 
                                            y_: y_batch, 
                                            keep_probability_: keep_prob, 
                                            l2_reg_: l2_reg})
        # checking errors
        if (cnt+1) % early_stopping_check_step == 0:
            print(np.round(time.time() - t, 2), end='s')
            t = time.time()
            preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
            preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

            train_err =  get_errors(preds_train, y_train_current)
            test_err =  get_errors(preds_test, y_test)
            print(' &', np.round(time.time() - t, 2), 's')
            #print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
            #data.append([al_iters] + train_err + test_err)
            # checking early stopping conditions
            if (test_err[0] > prev_test_error*(1 + early_stopping_window)) and (cnt > mandatory_uptrain_epochs):
                warnings += 1
                print('*'*warnings, end = '||')
                if warnings >= max_warnings:
                    print('$$$')
                    break
            else:
                warnings = 0
                prev_test_error = min(test_err[0], prev_test_error)
                save_path = saver.save(sess, "/tmp/2sgpnn_" + fname_identifier + ".ckpt")
                print("2step GPNN model saved in path: %s" % save_path)
            t = time.time()
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/tmp/2sgpnn_" + fname_identifier + ".ckpt")
    preds_train = sess.run(y, feed_dict={x: X_train_current, keep_probability_: 1})
    preds_test = sess.run(y, feed_dict= {x: X_test , keep_probability_: 1})

    train_err =  get_errors(preds_train, y_train_current)
    test_err =  get_errors(preds_test, y_test)
    print(epoch, np.round(train_err, 4), np.round(test_err, 4), end = '|')
    data.append([al_iters, sample_selection_time, uptraining_time] + train_err + test_err)
    lr, gs = sess.run([learning_rate, global_step])
    print('learning rate: {:.4E}, global step: {}'.format(lr, gs))
    datadf = pd.DataFrame(data, columns = data_columns).copy()
    datadf.to_csv('csvs/data_2sgpnn' + fname_identifier + '.csv', index = False)

#save_path = saver.save(sess, "/tmp/2sgpnn_" + fname_identifier + ".ckpt")
#print("2-step GPNN model saved in path: %s" % save_path)
