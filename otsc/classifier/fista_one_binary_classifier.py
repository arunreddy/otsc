from classifier.binary_classifier import BinaryClassifier
import numpy as np
import scipy.sparse as sparse


class FistaOneBinaryClassifier(BinaryClassifier):
    def __init__(self):
        BinaryClassifier.__init__(self)

    def prox_tlasso(self, f, t):
        _fmt = f[:] - t
        _fat = f[:] + t
        return np.maximum(0, _fmt) + np.minimum(0, _fat)

    def func_f(self, del_f):

        _fpf = np.add(self.f_prime, del_f)
        _fpf_sp = sparse.csr_matrix(_fpf)

        f1 = _fpf_sp.transpose() * self.L * _fpf_sp
        f1 = f1[0, 0]

        _fpfy = _fpf - self.y

        f2 = self.lambda_1 * np.dot(_fpfy.transpose(), _fpfy)
        f2 = f2[0, 0]

        return f1 + f2

    def grad_func_f(self, del_f):
        return 2 * (self.f_prime + del_f - self.y)

    def func_g(self, del_f):
        return np.abs(del_f).sum()

    def func_F(self, del_f):
        func_f = self.func_f(del_f)
        func_g = self.func_g(del_f)
        return func_f + func_g

    def func_G(self, x_del_f, y_del_f, L):
        q1 = self.func_f(y_del_f)
        x_y = np.subtract(x_del_f,y_del_f)
        q2 = np.dot(self.grad_func_f(y_del_f).transpose() , x_y)[0,0]
        q3 = (L / 2.) * np.dot(x_y.transpose() , x_y)[0,0]
        q4 = self.func_g(x_del_f)
        return q1 + q2 + q3 + q4

    def execute_algorithm(self):

        # Initialize parameters.

        # L_0 > 0
        l = 1.1

        # eta > 1
        eta = 2

        self.lambda_1 = 0.2
        self.lambda_2 = 0.2

        x_del_f = np.zeros(shape=(self.f_prime.shape[0], 1), dtype=float)
        y_del_f = x_del_f

        t = 1

        for k in range(2):

            print('\t>> Running iteration %d' % k)

            # Computing del_f_l
            i = 0
            l_cap = np.power(eta, i) * l

            f_F = self.func_F(self.prox_tlasso(y_del_f, l_cap))
            f_G = self.func_G(self.prox_tlasso(y_del_f, l_cap), y_del_f, l_cap)
            while f_F > f_G:
                print('\t\t>> Lipschitz iteration %d: %f %f' % (i, f_F, f_G))

                i += 1
                l_cap = np.power(eta, i) * l

                f_F = self.func_F(self.prox_tlasso(y_del_f, l_cap))
                f_G = self.func_G(self.prox_tlasso(y_del_f, l_cap), y_del_f, l_cap)

            l = np.power(eta, i) * l
            print('\t>> Lipschitz constant found after %d iterations. The value is %f' % (i, l))

            x_del_f_new = self.prox_tlasso(x_del_f, l)

            t_new = (1. + np.sqrt(1. + 4. * t * t)) / 2.

            y_del_f_new = x_del_f + ((t - 1) / t_new) * (x_del_f_new - x_del_f)

            # Update parameters
            t = t_new
            x_del_f = x_del_f_new
            y_del_f = y_del_f_new

    def exec(self, X, L, D, y, y_prime, f_prime, n_labeled=100):
        X_l, X_u, L, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u = self.train_test_split(X, L,
                                                                                                  y, y_prime,
                                                                                                  f_prime, n_labeled)

        self.S = L
        self.y_l = y_l
        self.y_u = y_u
        self.y_prime_l = y_prime_l
        self.y_prime_u = y_prime_u
        self.f_prime_l = f_prime_l
        self.f_prime_u = f_prime_u

        y_u_zeros = self.y_u.copy() * 0.0

        self.y = np.vstack((self.y_l, y_u_zeros))
        self.f_prime = np.vstack((f_prime_l, f_prime_u))
        self.y_prime = np.vstack((y_prime_l, y_prime_u))

        self.L = sparse.identity(self.S.shape[0]) - self.S

        self.m = self.y_l.shape[0]
        self.n = self.y_u.shape[0]

        self.lambda_1 = 0.2
        self.lambda_2 = 0.3

        self.execute_algorithm()
