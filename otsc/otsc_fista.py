"""
  OTSC FISTA implementation.
"""
import joblib
import numpy as np
import scipy.sparse as sparse
import logging


class OTSC_FISTA(object):
    def __init__(self, S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u):
        self.S = S
        self.D = D
        self.y_l = y_l
        self.y_u = y_u
        self.y_prime_l = y_prime_l
        self.y_prime_u = y_prime_u
        self.f_prime_l = f_prime_l
        self.f_prime_u = f_prime_u

        self.L = sparse.identity(self.S.shape[0]) - self.S

        m = self.y_l.shape[0]
        n = self.y_u.shape[0]

        self.L1 = self.L[:m, :m]
        self.L2 = self.L[m:, m:]
        self.L12 = self.L[:m, m:]

        print(self.L1.shape, self.L2.shape, self.L12.shape)

    def func_R1(self, del_f):
        if not type(del_f) is sparse.csr_matrix:
            print('>[ERR]: Given matrix is not a sparse matrix.')

        return np.abs(del_f.data).sum()

    def func_QL(self, del_f):
        fdf = self.f_prime_l + del_f
        A = fdf.transpose() * self.L1 * fdf
        B = self.lambda_1 * (fdf - y_l).transpose() * (fdf - y_l)
        return A + B

    def func_QU(self, del_f):
        fdf = self.f_prime_l + del_f
        A = fdf.transpose() * self.L * fdf
        B = self.lambda_1 * (fdf - y).transpose() * (fdf - y)
        return A + B

    def func_F(self, del_f, labeled=True):

        # Local and Global consistency
        if labeled:
            Q = self.func_QL(del_f)
        else:
            Q = self.func_QU(del_f)

        # Regularizer
        R1 = self.func_R1(del_f)
        R = self.lambda_2 * R1

        return Q[0, 0] + R

    def func_G(self, x_del_f, y_del_f, l_cap, labeled=True):

        fy = self.func_F(y_del_f)

        print(type(x_del_f),type(y_del_f))


        xmy = x_del_f - y_del_f
        gfy = self.grad_f(y_del_f)

        return fy + gfy.transpose()*xmy + (l_cap/2) * xmy.transpose()*xmy + self.func_R1(x_del_f)




        # return F(y) + np.dot((x - y), grad_f(y)) (l_prime / 2) * np.square(np.linalg.norm(x - y, ord=2)) + g(x)

        return 0.

    def grad_func_F(self):
        return self.f_prime_l

    def prox_tlasso(self, f, t):

        _fmt = f.copy()
        _fat = f.copy()

        _fmt.data = _fmt.data - t
        _fat.data = _fat.data + t
        _fmt.eliminate_zeros()
        _fat.eliminate_zeros()

        _zeros = f.multiply(0.)

        fmt = _fmt.multiply(_fmt >= _zeros)
        fat = _fat.multiply(_fat <= _zeros)

        prox = fmt + fat
        prox = prox.eliminate_zeros()

        return prox

    def prox_t2norm(self, f):
        pass

    def prox_telasticnet(f):
        pass

    def execute_algorithm(self):

        # Initialize parameters.

        # L_0 > 0
        l_l = 1
        l_u = 1

        # eta > 1
        eta = 2

        self.lambda_1 = 0.2
        self.lambda_2 = 0.2

        x_del_f_l = sparse.csr_matrix((f_prime_l.shape[0], 1), dtype=float)
        x_del_f_u = sparse.csr_matrix((f_prime_u.shape[0], 1), dtype=float)

        y_del_f_l = x_del_f_l
        y_del_f_u = x_del_f_u

        t_l = 1
        t_u = 1

        for k in range(2):
            # Computing del_f_l
            i = 0
            l_l_cap = np.power(eta,i) * l_l
            while self.func_F(x_del_f_l) > self.func_G(self.prox_tlasso(y_del_f_l, t_l), y_del_f_l,l_l_cap):
                i += 1
                l_l_cap = np.power(eta, i) * l_l
                if i > 10:
                    break


# def otsc_fista(S, D, y_l, y_prime_l, y_prime_u, f_prime_l, f_prime_u):
#
#
#
#
#     for k in range(2):
#
#
#
#
#
#
#         # Computing del_f_u
#
#
#     f_prime, del_f, y = combine_l_u(f_prime_l, f_prime_u, del_f_l, del_f_u, y_l, y_u)
#     # F(f_prime, del_f, S, lambda_1, y)
#
#
#
#     # x_0 = np.zeros(1)
#     # y_1 = x_0
#     # t_1 = 1
#     # x = x_0
#     #
#     # for k in range(2):
#     #
#     #     ik = 0
#     #     l_prime = np.power(eta, ik) * l
#     #
#     #     # Compute the Lipschitz constant L
#     #     while F(x) > ():
#     #         ik += 1
#     #         l_prime = np.power(eta, ik) * l
#     #
#     #     l = l_prime
#     #
#     #     x_next = prox_tlasso(y, l)
#     #
#     #     t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
#     #
#     #     y_next = x_next + ((t - 1) / t_next) * (x_next - x)
#     #
#     #     x = x_next
#     #     y = y_next
#     #     t = t_next


if __name__ == '__main__':
    S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u = joblib.load('features.dat')

    print('S', S.shape)
    print('D', D.shape)
    print('y_l', len(y_l))
    print('y_u', len(y_u))
    print('y_prime_l', len(y_prime_l))
    print('y_prime_u', len(y_prime_u))

    S = sparse.csr_matrix(S)
    y_l = sparse.csr_matrix(y_l).transpose()
    y_u = sparse.csr_matrix(y_u).transpose()
    f_prime_l = sparse.csr_matrix(f_prime_l).transpose()
    f_prime_u = sparse.csr_matrix(f_prime_u).transpose()

    print('y', y_l.shape, y_u.shape)
    print('f_prime', f_prime_l.shape, f_prime_u.shape)

    obj = OTSC_FISTA(S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u)
    obj.execute_algorithm()
