#!/usr/bin/env python
# coding=utf-8
"""
This file runs all the experiments.
"""
from __future__ import division, print_function, unicode_literals

import logging
import os

import joblib
import numpy as np
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver

from classifier.classifier_utils import train_test_split_with_fixed_test
from classifier.classifiers import Classifiers
from dataset import Imdb, AmazonReviewsBinary, AmazonFineFoodReviews
from feat.features import FeaturesGenerator

logger = logging.getLogger()

# ============== Ingredient 0: settings ==============================
settings_ingredient = Ingredient("settings")


@settings_ingredient.config
def cfg1():
    verbose = True
    data_home = '/media/d2/data-tmp/otsc/data'


# ============== Ingredient 1: dataset ==============================
dataset_ingredient = Ingredient("dataset", ingredients=[settings_ingredient])


@dataset_ingredient.config
def cfg2(settings):
    v = not settings['verbose']
    base = '/home/sacred/'
    data_set_name = 'amazon_fine_foods'
    weighted = False
    n_total = 100
    sim = 'cosine'

    labeled_examples_start = 4
    labeled_examples_end = 100
    labeled_examples_step = 4
    n_random_states = 3


@dataset_ingredient.capture
def load_data(settings, data_set_name, n_total, weighted, sim):
    # If, a joblib file exists, return it.
    if weighted:
        joblib_file = os.path.join(settings['data_home'], data_set_name + str(n_total) + '_weighted_%s.dat' % sim)
    else:
        joblib_file = os.path.join(settings['data_home'], data_set_name + str(n_total) + '_%s.dat' % sim)

    if os.path.exists(joblib_file):
        logger.info('Returning from the joblib cache from file %s' % joblib_file)
        return joblib.load(joblib_file)

    # Else, create one, save and return it.
    if data_set_name == 'imdb':
        dataset = Imdb()
    elif data_set_name == 'amazon_fine_foods':
        dataset = AmazonFineFoodReviews()
    elif data_set_name == 'amazon_binary':
        dataset = AmazonReviewsBinary()
    else:
        logger.error('> Specify the dataset.')

    # Generate the features.
    df_pos, df_neg = dataset.load_data(n_total)

    feat_generator = FeaturesGenerator()
    X, L, D, y, y_prime, f_prime = feat_generator.generate_features(df_pos, df_neg, n_total, feat_type='tf-idf',
                                                                    sim=sim)

    if weighted:
        f_prime = np.multiply(y_prime, f_prime)

    logger.info('> Dumping the data into %s' % joblib_file)

    # data = Data(X=X, y=y, yp=y_prime, fp=f_prime, L=L, D=D)
    joblib.dump([X, L, D, y, y_prime, f_prime], joblib_file, compress=5)

    return X, L, D, y, y_prime, f_prime


def run_experiments(_run, X, L, D, y, y_prime, f_prime, n_iterations=100, n_random_states=2):
    data_set_name = _run.config['dataset']['data_set_name']
    n_random_states = _run.config['dataset']['n_random_states']
    start = _run.config['dataset']['labeled_examples_start']
    end = _run.config['dataset']['labeled_examples_end']
    step = _run.config['dataset']['labeled_examples_step']

    for random_state in range(n_random_states):
        logger.info('Running all the algorithms for the random state %d' % random_state)
        for n_labeled in range(start, end + 1, step):
            X_l, X_u, l, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u = train_test_split_with_fixed_test(X, L, y, y_prime,
                                                                                                 f_prime,
                                                                                                 n_labeled,
                                                                                                 9000,
                                                                                                 rand=random_state)

            classifiers_list = ['fista_lasso_boost_svr', 'fista_lasso_boost', 'svm', 'svm_adversarial', 'xgb']
            # ['xgb', 'svm', 'fista_lasso_boost', 'fista_lasso_boost_svr']  #
            Classifiers.execute_classifiers(_run, X_l, X_u, l, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, classifiers_list,
                                            random_state, n_labeled)


# ============== Experiment ==========================================
ex = Experiment('OTSC', ingredients=[dataset_ingredient])
ex.observers.append(MongoObserver.create())


#
# @ex.pre_run_hook
# def pre_hook(_run):
#
#
#
@ex.post_run_hook
def post_hook(_run):
    logger.info('Finished running the algorithms.')
    print(_run.config)
    print(_run)
    print(_run.meta_info)
    print(_run.info)

    # plt.figure(figsize=(8.5, 4))
    # plot(_run._id, '_cosine')
    # plt.grid(linestyle='dotted')
    # plt.legend()
    # plt.tight_layout()
    # plt.title('Amazon Fine Food reviews dataset')
    # plt.savefig('figure.png', dpi=600)
    # # plt.show()
    #


@ex.automain
def main(_run):
    data_set_name = _run.config['dataset']['data_set_name']
    # Load the dataset.
    X, L, D, y, y_prime, f_prime = load_data(data_set_name=data_set_name, n_total=5000, weighted=True,
                                             sim='cosine')

    # den = np.max(f_prime) - np.min(f_prime)
    # # f_prime = f_prime - np.min(f_prime)
    # #
    # f_prime = f_prime/den

    #
    # hist = np.histogram(f_prime,bins=20,range=[-1,1])
    # print(hist)

    # import matplotlib.pyplot as plt
    #
    # plt.hist(f_prime,bins=20)
    # plt.show()
    # # Execute the experiments.
    run_experiments(_run, X, L, D, y, y_prime, f_prime)
