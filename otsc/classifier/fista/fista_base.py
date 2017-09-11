"""
Base class for the Fast Iterative Shrinkage-Thresholidng Algorithm
"""

import logging

import numpy as np
import scipy.sparse as sparse
from otsc.classifier.base_classifier import BaseClassifier

class FISTABase(BaseClassifier):

    def __init__(self):
        self.logger = logging.getLogger(self.__name__)

        self.prox_lambda = 0.01

    def prox_tlasso(self, del_f):
        _fmt = del_f[:] - self.prox_lambda
        _fat = del_f[:] + self.prox_lambda
        return np.maximum(0, _fmt) + np.minimum(0, _fat)

    def func_f(self, L, y, f_p, del_f, lambda_1):
        _fpf = np.add(f_p, del_f)
        _fpf_sp = sparse.csr_matrix(_fpf)
        f1 = _fpf_sp.transpose() * L * _fpf_sp
        f1 = f1[0, 0]

        _fpfy = _fpf - y

        f2 = lambda_1 * np.dot(_fpfy.transpose(), _fpfy)
        f2 = f2[0, 0]

        return 0.5 * (f1 + f2)

    def grad_func_f(self, L, y, f_p, del_f, lambda_1):
        _fpf = np.add(f_p, del_f)
        _fpf_sp = sparse.csr_matrix(_fpf)
        f1 = L * _fpf_sp
        f1 = f1.todense()

        _fpfy = _fpf - y
        f2 = lambda_1 * _fpfy

        return np.add(f1, f2)

    def func_g(self, del_f):
        return np.abs(del_f).sum()

    def p_L(self, L, y, f_p, del_f, lambda_1, l_cap=1):
        if l_cap == 0:
            self.logger.error('> Error in L value. L value is 0')
        t = 1 / l_cap
        return del_f - t * self.grad_func_f(L, y, f_p, del_f, lambda_1)

    def func_F(self, L, y, f_p, del_f, lambda_1, lambda_2):
        func_f = self.func_f(L, y, f_p, del_f, lambda_1)
        func_g = self.func_g(del_f)
        return func_f + lambda_2 * func_g

    def func_G(self, L, y, f_p, x_del_f, y_del_f, lambda_1, lambda_2, l_cap):
        q1 = self.func_f(L, y, f_p, y_del_f, lambda_1)
        x_y = np.subtract(x_del_f, y_del_f)
        q2 = np.dot(self.grad_func_f(L, y, f_p, y_del_f, lambda_1).transpose(), x_y)[0, 0]
        q3 = (l_cap / 2.) * np.dot(x_y.transpose(), x_y)[0, 0]
        q4 = lambda_2 * self.func_g(x_del_f)
        return q1 + q2 + q3 + q4


    def compute_accuracy(self, y, f):
        f_acc_p = 0.
        f_acc_n = 0.
        f_acc_z = 0.
        n = y.shape[0]
        for i in range(n):
            if y[i] * f[i] > 0:
                if f[i] > 0:
                    f_acc_p += 1.
                elif f[i] < 0:
                    f_acc_n += 1.
                else:
                    f_acc_z += 1.

        return f_acc_p, f_acc_n