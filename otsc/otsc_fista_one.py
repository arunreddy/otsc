import scipy.sparse as sparse
import numpy as np
import joblib


class OtscFistaOne(object):
    def __init__(self, S, D, y, y_prime, f_prime, n_labeled):
        self.S = S
        self.D = D
        self.y_l = y_l
        self.y_u = y_u
        self.y_prime_l = y_prime_l
        self.y_prime_u = y_prime_u
        self.f_prime_l = f_prime_l
        self.f_prime_u = f_prime_u

        y_u_zeros = self.y_u.copy() * 0.0

        self.y = np.hstack((self.y_l, y_u_zeros))
        self.f_prime = np.hstack((f_prime_l, f_prime_u))
        self.y_prime = np.hstack((y_prime_l, y_prime_u))

        self.L = sparse.identity(self.S.shape[0]) - self.S

        self.m = self.y_l.shape[0]
        self.n = self.y_u.shape[0]

        self.lambda_1 = 0.2
        self.lambda_2 = 0.3



if __name__ == '__main__':
    S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u = joblib.load('../data/features.dat')

    print('S', S.shape)
    print('D', D.shape)
    print('y_l', len(y_l))
    print('y_u', len(y_u))
    print('y_prime_l', len(y_prime_l))
    print('y_prime_u', len(y_prime_u))

    S = sparse.csr_matrix(S)

    y_l = np.asarray(y_l, dtype=float)
    y_u = np.asarray(y_u, dtype=float)

    f_prime_l = np.asarray(f_prime_l, dtype=float)
    f_prime_u = np.asarray(f_prime_u, dtype=float)

    y_prime_l = np.asarray(y_prime_l, dtype=float)
    y_prime_u = np.asarray(y_prime_u, dtype=float)

    print('y', y_l.shape, y_u.shape)
    print('f_prime', f_prime_l.shape, f_prime_u.shape)
    print('y_prime', y_prime_l.shape, y_prime_u.shape)

    obj = OtscFistaOne(S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u)
    obj.execute_algorithm()
