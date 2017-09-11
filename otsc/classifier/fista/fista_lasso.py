import numpy as np

from otsc.classifier.fista.fista_base import FISTABase


class FISTALasso(FISTABase):
    def __init__(self, _run, n_labeled):
        self._run = _run
        # Load configuration.
        self.prox_lambda = 0.01

        self.lambda_1 = 0.9
        self.lambda_2 = 0.9
        self.n_iterations = 100
        self.n_labeled = n_labeled

    def __str__(self):
        return 'FISTA Lasso with lambda_1[%0.2f], lambda_2[%0.2f], n_labeled[%d]' % (self.lambda_1,
                                                                                self.lambda_2,
                                                                                self.n_labeled)

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

        for k in range(iterations):

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

    def exec(self, X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, random_state):
        data_set_name = self._run.config['dataset']['data_set_name']

        y_u_zeros = y_u.copy() * 0.0

        y = np.vstack((y_l, y_u_zeros))
        f_p = np.vstack((f_p_l, f_p_u))
        y_p = np.vstack((y_p_l, y_p_u))

        m = y_l.shape[0]
        n = y_u.shape[0]

        # Compute the stanford values.
        s_acc = self.compute_accuracy(y_u, y_p_u)
        y_p_u_cen = np.copy(y_p_u)
        y_p_u_cen = y_p_u_cen - np.mean(y_p_u_cen)
        s_acc_cen = self.compute_accuracy(y_u, y_p_u_cen)

        self._run.log_scalar(
            '%s.fista_lasso.s_acc.%0.2f.%0.2f.%0.2d' % (data_set_name, self.lambda_1, self.lambda_2, random_state),
            s_acc, step=m)
        self._run.log_scalar(
            '%s.fista_lasso.s_acc_cen.%0.2f.%0.2f.%0.2d' % (data_set_name, self.lambda_1, self.lambda_2, random_state),
            s_acc_cen, step=m)

        f_val, l_val, x_del_f = self.execute_algorithm(L, y, f_p, self.lambda_1, self.lambda_2, self.n_iterations)



        f_predicted = np.add(f_p_u, x_del_f[m:])



        f_predicted_center = f_predicted - f_predicted.mean()

        f_acc = self.compute_accuracy(y_u, f_predicted)
        f_acc_cen = self.compute_accuracy(y_u, f_predicted_center)
        self._run.log_scalar(
            '%s.fista_lasso.f_acc.%0.2d.%0.2f.%0.2f' % (data_set_name, random_state, self.lambda_1, self.lambda_2),
            f_acc, step=m)
        self._run.log_scalar(
            '%s.fista_lasso.f_acc_cen.%0.2d.%0.2f.%0.2f' % (data_set_name, random_state, self.lambda_1, self.lambda_2),
            f_acc_cen, step=m)

        return None
