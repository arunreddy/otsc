import logging

from otsc.classifier.baseline import SVMClassifier, XGBClassifier, SVMAdversarial
from otsc.classifier.fista import FISTALasso, FISTALassoBoost, FISTALassoBoostInc
import numpy as np

class Classifiers(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def execute_classifiers(_run, X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, classifiers, random_state,
                            n_labeled):
        """

        :param _run: handle to experiment run object
        :param data_train_test: data_train_test tuple
        :param classifiers: list of classifiers to run.
        :return: None
        """

        for _classifier in classifiers:

            if _classifier == 'fista_lasso':
                clf = FISTALasso(_run, n_labeled)

            elif _classifier == 'fista_lasso_boost_svr':
                clf = FISTALassoBoost(_run, n_labeled, boost_type='svr')

            elif _classifier == 'fista_lasso_boost':
                clf = FISTALassoBoost(_run, n_labeled)

            elif _classifier == 'svm':
                clf = SVMClassifier(_run, n_labeled)

            elif _classifier == 'xgb':
                clf = XGBClassifier(_run, n_labeled)

            elif _classifier == 'fista_lasso_boost_inc_svr':
                clf = FISTALassoBoostInc(_run, n_labeled, boost_type='svr')

            elif _classifier == 'svm_adversarial':
                c = SVMAdversarial(_run, n_labeled)
                ayl, ayu = c.exec(X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, random_state)
                f_p_l = np.multiply(f_p_l, ayl)
                f_p_u = np.multiply(f_p_u, ayu)
                clf = FISTALassoBoost(_run, n_labeled, boost_type='svr', tag='adv')

            elif _classifier == 'fista_lasso_boost_inc':
                clf = FISTALassoBoostInc(_run, n_labeled)
            else:
                print('Unknown classifier called.')

            print('\t>', clf)
            clf.exec(X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, random_state)
