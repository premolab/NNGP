import random
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.nngp import NNGP
from uncertainty_estimator.mcdue import MCDUE
from uncertainty_estimator.random_estimator import RandomEstimator
from sample_selector.eager import EagerSampleSelector
from oracle.identity import IdentityOracle
from al_trainer import ALTrainer

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False




def build_estimator(name, model):
    if name == 'nngp':
        estimator = NNGP(model)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def run_experiment(config):
    """
    Run active learning for the 10D rosenbrock function data
    It starts from small train dataset and then extends it with points from pool

    We compare three sampling methods:
    - Random datapoints
    - Points with highest uncertainty by MCDUE
    - Points with highest uncertainty by NNGP (proposed method)
    """
    sess = None
    rmses = {}

    for estimator_name in config['estimators']:
        print("\nEstimator:", estimator_name)

        # load data
        X_train, y_train, X_val, y_val, _, _, X_pool, y_pool = RosenData(
            config['n_train'], config['n_val'], config['n_test'], config['n_pool'], config['n_dim']
        ).dataset(use_cache=config['use_cached_data'])

        # Init neural network & tf session
        tf.reset_default_graph()
        if config['random_seed'] is not None:
            tf.set_random_seed(config['random_seed'])
            np.random.seed(config['random_seed'])
            random.seed(config['random_seed'])

        model = MLP(ndim=config['n_dim'], layers=config['layers'])

        if sess is not None and not sess._closed:
            sess.close()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        model.set_session(sess)

        # Init other parts
        estimator = build_estimator(estimator_name, model)  # to estimate uncertainties
        oracle = IdentityOracle(y_pool)  # generate y for X from pool
        sampler = EagerSampleSelector(oracle)  # sample X and y from pool by uncertainty estimations

        trainer = ALTrainer(
            model, estimator, sampler, oracle, config['al_iterations'],
            config['update_sample_size'], verbose=config['verbose'])
        rmses[estimator_name] = trainer.train(X_train, y_train, X_val, y_val, X_pool)

    visualize(rmses)


def visualize(rmses):
    print(rmses)
    plt.figure(figsize=(12, 9))
    plt.xlabel('Active learning iteration')
    plt.ylabel('Validation RMSE')
    for estimator_name, rmse in rmses.items():
        plt.plot(rmse, label=estimator_name, marker='.')

    plt.title('RMS Error by active learning iterations')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    config = {
        'random_seed': 42,
        'n_dim': 10,
        'n_train': 200,
        'n_val': 200,
        'n_test': 200,
        'n_pool': 1000,
        'layers': [128, 64, 32],
        'update_sample_size': 100,
        'al_iterations': 1,
        'use_cached_data': True,
        'verbose': False,
        'estimators': ['nngp', 'mcdue', 'random']
    }

    run_experiment(config)


