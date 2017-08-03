import scipy.sparse as sparse
import numpy as np
import joblib


class OtscFistaOne(object):
    def __init__(self, S, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u):
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

    def prox_tlasso(self, f, t):
        _fmt = f[:] - t
        _fat = f[:] + t
        return np.maximum(0, _fmt) + np.minimum(0, _fat)

    def func_f(self, del_f):

        _fpf = self.f_prime + del_f
        print(_fpf.shape)

        _fpf_sp = sparse.csr_matrix(_fpf).transpose()

        print(_fpf_sp.shape,self.L.shape)

        f1 = _fpf_sp.transpose()*self.L*_fpf_sp
        f1 = f1[0,0]

        _fpfy = _fpf - self.y
        f2 = self.lambda_1* _fpfy.transpose()*_fpfy

        return f1 + f2


    def grad_func_f(self, del_f):
        return 2*(self.f_prime + del_f - self.y)

    def func_g(self, del_f):
        return np.abs(del_f).sum()

    def func_F(self, del_f):
        return self.func_f(del_f) + self.func_g(del_f)


    def func_Q(self, x_del_f, y_del_f, L):
        q1 = self.func_f(y_del_f)
        x_y = x_del_f - y_del_f
        q2 = self.grad_func_f(y_del_f).transpose()*x_y
        q3 = (L/2.)*x_y.transpose()*x_y
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

            print('\t>> Running iteration %d'%k)

            # Computing del_f_l
            i = 0
            l_cap = np.power(eta, i) * l

            f_F = self.func_F(x_del_f)
            f_G = self.func_G(self.prox_tlasso(y_del_f, l_cap), y_del_f, l_cap)
            while  f_F >= f_G :

                print('\t\t>> Lipschitz iteration %d: %f %f'%(i,f_F,f_G))

                i += 1
                l_cap = np.power(eta, i) * l

                f_F = self.func_F(x_del_f)
                f_G = self.func_G(self.prox_tlasso(y_del_f, l_cap), y_del_f, l_cap)


            l = np.power(eta, i) * l
            print('\t>> Lipschitz constant found after %d iterations. The value is %f'%(i,l))

            x_del_f_new = self.prox_tlasso(x_del_f,l)

            t_new = (1. + np.sqrt(1. + 4. * t * t))/2.

            y_del_f_new = x_del_f + ((t-1)/t_new)*(x_del_f_new - x_del_f)


            # Update parameters
            t = t_new
            x_del_f = x_del_f_new
            y_del_f = y_del_f_new






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
