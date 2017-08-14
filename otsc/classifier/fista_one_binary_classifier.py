from classifier.binary_classifier import BinaryClassifier
import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import joblib


class FistaOneBinaryClassifier(BinaryClassifier):
    def __init__(self, X, L, D, y, y_prime, f_prime, prox_lambda=0.01):
        BinaryClassifier.__init__(self)

        self.X = X
        self.L = L
        self.D = D
        self.y = y
        self.y_prime = y_prime
        self.f_prime_r = f_prime
        self.f_prime = y_prime  # np.multiply(f_prime, y_prime)

        self.prox_lambda = prox_lambda

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
            print('> Error in L value. L value is 0')
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

    def execute_algorithm(self, L, y, f_p, lambda_1, lambda_2, iterations=100):

        # Initialize parameters.

        # L_0 > 0
        l = 1.1

        # eta > 1
        eta = 2

        x_del_f = np.zeros(shape=(f_p.shape[0], 1), dtype=float)
        y_del_f = x_del_f

        t = 1

        f_val = []
        l_val = []
        for k in tqdm(range(iterations), desc='[λ_1 :%0.2f and λ_2:%0.2f]'%(lambda_1,lambda_2)):

            # Computing del_f_l
            i = 0
            l_cap = np.power(eta, i) * l

            f_F = self.func_F(L, y, f_p, self.prox_tlasso(self.p_L(L, y, f_p, y_del_f, lambda_1, l_cap)), lambda_1,
                              lambda_2)
            f_G = self.func_G(L, y, f_p, self.prox_tlasso(self.p_L(L, y, f_p, y_del_f, lambda_1, l_cap)), y_del_f,
                              lambda_1,
                              lambda_2, l_cap)
            while f_F > f_G:
                # print('\t\t>> Lipschitz iteration %d: %f %f' % (i, f_F, f_G))

                i += 1
                l_cap = np.power(eta, i) * l

                f_F = self.func_F(L, y, f_p, self.prox_tlasso(self.p_L(L, y, f_p, y_del_f, lambda_1, l_cap)), lambda_1,
                                  lambda_2)
                f_G = self.func_G(L, y, f_p, self.prox_tlasso(self.p_L(L, y, f_p, y_del_f, lambda_1, l_cap)), y_del_f,
                                  lambda_1,
                                  lambda_2, l_cap)

            l = np.power(eta, i) * l
            # print(
            #     '\t>> Lipschitz constant found after %d iterations. [L: %0.2f, F: %0.2f, G: %0.2f]' % (i, l, f_F, f_G))

            x_del_f_new = self.prox_tlasso(self.p_L(L, y, f_p, x_del_f, lambda_1, l))

            t_new = (1. + np.sqrt(1. + 4. * t * t)) / 2.

            y_del_f_new = x_del_f + ((t - 1) / t_new) * (x_del_f_new - x_del_f)

            # Update parameters
            t = t_new
            x_del_f = x_del_f_new
            y_del_f = y_del_f_new

            f_val.append(self.func_F(L, y, f_p, x_del_f, lambda_1, lambda_2))
            l_val.append(l)

        return f_val, l_val, x_del_f


    def exec(self, X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, lambda_1, lambda_2, iterations=100):
        y_u_zeros = y_u.copy() * 0.0

        y = np.vstack((y_l, y_u_zeros))
        f_p = np.vstack((f_p_l, f_p_u))
        y_p = np.vstack((y_p_l, y_p_u))

        m = y_l.shape[0]
        n = y_u.shape[0]

        f_val, l_val, x_del_f = self.execute_algorithm(L, y, f_p, lambda_1, lambda_2, iterations)

        f_predicted = np.add(f_p_u, x_del_f[m:])

        # Stanford accuracy
        s_acc = 0.
        for i in range(n):
            if y_u[i] * y_p_u[i] > 0:
                s_acc += 1.

        #
        f_acc = 0.
        for i in range(n):
            if y_u[i] * f_predicted[i] > 0:
                f_acc += 1.


        s_acc = s_acc/n
        f_acc = f_acc/n

        return f_val, l_val, x_del_f, s_acc, f_acc

    # def analyze_data(self, lambda_1, lambda_2, iterations=100):
    #
    #     lambda_1 = 0.4
    #     lambda_2 = 0.6
    #
    #     # Split train and test data sets
    #     X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u = self.train_test_split(self.X,
    #                                                                               self.L,
    #                                                                               self.y,
    #                                                                               self.y_prime,
    #                                                                               self.f_prime,
    #                                                                               n_labeled)
    #
    #     y_u_zeros = y_u.copy() * 0.0
    #
    #     y = np.vstack((y_l, y_u_zeros))
    #     f_p = np.vstack((f_p_l, f_p_u))
    #     y_p = np.vstack((y_p_l, y_p_u))
    #
    #     m = y_l.shape[0]
    #     n = y_u.shape[0]
    #
    #
    #     # results.append([lambda_1, lambda_2, s_acc, f_acc, np.sum(x_del_f)])
    #
    #     # plt.plot(list(range(len(f_val))), f_val, c='k')
    #
    #     f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #
    #     results = []
    #     for lambda_1 in np.arange(0.3, 1.5, 0.1):
    #         for lambda_2 in np.arange(0.1, 1.5, 0.1):
    #
    #             f_val, l_val, x_del_f = self.execute_algorithm(L, y, f_p, lambda_1, lambda_2, iterations)
    #
    #             f_predicted = np.add(f_p_u, x_del_f[m:])
    #
    #             # Stanford accuracy
    #             s_acc = 0
    #             for i in range(n):
    #                 if y_u[i] * y_p_u[i] > 0:
    #                     s_acc += 1
    #
    #             #
    #             f_acc = 0
    #             for i in range(n):
    #                 if y_u[i] * f_predicted[i] > 0:
    #                     f_acc += 1
    #
    #             print([lambda_1, lambda_2, s_acc, f_acc, np.sum(x_del_f)])
    #
    #             ax1.plot(list(range(len(l_val))), f_val, c='g')
    #             ax2.plot(list(range(len(f_val))), l_val, c='g')
    #             ax3.plot(list(range(len(f_val))), np.divide(1., l_val), c='g')
    #
    #
    #     plt.show()
        #
        #         # Execute algorithm
        #         f_val, x_del_f = self.execute_algorithm(L, y, f_p, lambda_1, lambda_2)
        #
        #         f_predicted = np.add(f_p_u, x_del_f[m:])
        #
        #         # Stanford accuracy
        #         s_acc = 0
        #         for i in range(n):
        #             if y_u[i] * y_p_u[i] > 0:
        #                 s_acc += 1
        #
        #         #
        #         f_acc = 0
        #         for i in range(n):
        #             if y_u[i] * f_predicted[i] > 0:
        #                 f_acc += 1
        #
        #         print([lambda_1, lambda_2, s_acc, f_acc, np.sum(x_del_f)])
        #         results.append([lambda_1, lambda_2, s_acc, f_acc, np.sum(x_del_f)])
        #
        #         # for i in range(n):
        #         #     print("%0.2f %0.2f  %0.2f %0.2f"%(y_u[i],y_p_u[i],f_p_u[i], f_predicted[i]))
        #         #
        #         #
        #
        # # joblib.dump(results, '/tmp/results_l1_l2.dat', compress=5)
