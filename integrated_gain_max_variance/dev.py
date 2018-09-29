
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import time
import sys
import time
import tqdm
from scipy.optimize import rosen

get_ipython().run_line_magic('matplotlib', 'inline')


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


# In[3]:


df = np.random.uniform(size=(4000, 10))
targets = rosen(df.T)

train_num = 500
thres = 3500
thres2 = 4000


X_train = df[:train_num]
y_train = targets[:train_num][:, None]
X_pool = df[train_num:thres]
y_pool = targets[train_num:thres][:, None]
X_test = df[thres:thres2]
y_test = targets[thres:thres2][:, None]
print('train shapes:', X_train.shape, y_train.shape)
print('pool shapes:', X_pool.shape, y_pool.shape)
print('test shapes:', X_test.shape, y_test.shape)


# In[4]:


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

def get_stds(X):
    stds = np.zeros((X.shape[0], T), dtype = float)
    for cnt_ in range(T):
        stds[:, cnt_] = np.ravel(sess.run(y, feed_dict={x: X, 
                                                        keep_probability_: .5}))
    return stds


# In[5]:


ndim = X_train.shape[1]
# layers = [64,32]
layers = [64,64,32]

learning_rate_decay = .97
start_learning_rate = 8e-4
learning_rate_schedule_epochs = 50000


# $X$: batch_size $\times$ dim 
# 
# $W$: dim $\times$ 1
# 
# 
# output: batch_size $\times$ 1

# In[6]:


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


# In[7]:


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


# In[8]:


batch_size = 500
init_epochs = 50000
keep_prob = .95
l2_reg = 5e-5

al_steps = 20
uptrain_epochs = 1000000
mandatory_uptrain_epochs = 10000
sample_each_step = 250
T = 25

early_stopping_window = .03
max_warnings = 3
early_stopping_check_step = 100

gpnn_max_train = 1000
diag_eps = .01


# In[9]:


X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()


# # # Initial_training

# In[10]:


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


# In[10]:


fname_identifier = "rosenbrock_exp"
save_path = saver.save(sess, "/Users/romaushakov/Desktop/diploma/init_" + fname_identifier + ".ckpt")
print("Init model saved in path: %s" % save_path)


# In[10]:


fname_identifier = "rosenbrock_exp"
saver.restore(sess, "/Users/romaushakov/Desktop/diploma/init_" + fname_identifier + ".ckpt")
print("Init model restored")


# In[11]:


X_train_current = X_train.copy()
y_train_current = y_train.copy()
X_pool_current = X_pool.copy()
y_pool_current = y_pool.copy()


# In[12]:


def compute_block_inv(A_inv, B, C, D):
    H = D - C.dot(A_inv).dot(B)
    H_inv = 1./ H
    a00 = A_inv + H_inv * A_inv.dot(B).dot(C).dot(A_inv)
    a01 = -A_inv.dot(B) * H_inv
    a10 = -H_inv * C.dot(A_inv)
    a11 = H_inv
    
    return np.block([[a00, a01.reshape(-1, 1)],
                    [a10.reshape((1, -1)), np.array(a11).reshape((1, 1))]])


# In[27]:


gpnn_max_train = 100
points_to_integrate = 500

print('='*40)
print('Integral-based ALGO')
print('='*40)

for al_iters in range(al_steps):
    # 1) get MCDUEs
    t = time.time()
    print('Starting iteration #', al_iters)
    random_train_inds = np.random.permutation(range(len(X_train_current)))[:gpnn_max_train]
    random_train_samples = X_train_current[random_train_inds]
    
    train_and_pool_samples = np.concatenate([random_train_samples, X_pool_current])    
    stds = get_stds(train_and_pool_samples)
    
    K_train_cov = np.cov(stds[:gpnn_max_train, :], ddof = 0)
    K_train_cov_inv = np.linalg.inv(K_train_cov + diag_eps * np.eye(gpnn_max_train))
    
    minimums = random_train_samples.min(axis=0)
    maximums = random_train_samples.max(axis=0)
    
    ### vs are points for integral
    vs = np.random.uniform(minimums, maximums,
                           size=(points_to_integrate, 
                                 random_train_samples.shape[1]))
    
    # get mcdues for random vs
    y_vs = get_stds(vs)

    ### sigma(v | X) for each v in vs
    sigmas = []
    for cnt_ in range(len(vs)):
        vs_sample = y_vs[cnt_, :]
        Q = simple_cov(stds[:gpnn_max_train], vs_sample)[:, None]
        KK = np.var(vs_sample)
        sigma = KK - np.dot(np.dot(Q.T, K_train_cov_inv), Q)[0][0]
        sigmas.append(np.sqrt(sigma))
    

    # for each x in X_pool_current:
    # we count \int sigma(v|X) - sigma(v|X+x_from_pool) dv
    diffs_integral = np.zeros(X_pool_current.shape[0])
    
    ### extend cov matrix 
    new_K_cov = np.zeros((gpnn_max_train + 1, gpnn_max_train + 1))
    new_K_cov[:gpnn_max_train, :gpnn_max_train] = K_train_cov
    
    ### loop over pool data
    for x_cnt_ in tqdm.tqdm(range(len(X_pool_current))):
        
        # stds was recieved for train_and_pool_samples 
        # and train_pool_sample = np.concatenate([random_train_sample, X_pool_current])
        # and random_train_samples.shape[0] = gpnn_max_train. So
        
        
        # extend cov matrix
        # we don't recalculate all cov matrix
        # we only add one row 
        pool_sample = stds[(gpnn_max_train + x_cnt_), :]
        Q = simple_cov(stds[:gpnn_max_train, :], pool_sample)[:, None]
        Q = Q.ravel()
        new_K_cov[-1, :-1] = Q
        new_K_cov[:-1, -1] = Q
        new_K_cov[-1, -1] = np.var(pool_sample)
        new_K_cov_inv = compute_block_inv(K_train_cov_inv,
                                          Q.reshape((-1, 1)),
                                          Q.reshape((1, -1)), 
                                          np.var(pool_sample) + diag_eps)
        
        
        indices = list(range(gpnn_max_train)) + [gpnn_max_train + x_cnt_]

        ### count sigma(v | X + x_from_pool) with extended 
        ### cov matrix for each v in vs
        extended_sigmas = []
        for cnt_ in range(len(y_vs)):
            vs_sample = y_vs[cnt_, :]
            Q = simple_cov(stds[indices], vs_sample)[:, None]
            KK = np.var(vs_sample)
            sigma = KK + diag_eps - np.dot(np.dot(Q.T, new_K_cov_inv), Q)[0][0]
            extended_sigmas.append(sigma)
        

        current_diff = np.array(sigmas) - np.array(extended_sigmas)
        diffs_integral[x_cnt_] = current_diff.sum()

        
    inds = np.argsort(diffs_integral)[::-1][:sample_each_step]

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
            # data.append([al_iters] + train_err + test_err)
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
                save_path = saver.save(sess, "/Users/romaushakov/Desktop/diploma/init_" + fname_identifier + ".ckpt")
                print("MCDUE model saved in path: %s" % save_path)
            t = time.time()
            
    print('NN uptrained')
    uptraining_time = time.time() - t_big
    saver.restore(sess, "/Users/romaushakov/Desktop/diploma/init_" + fname_identifier + ".ckpt")
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

