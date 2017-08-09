import numpy as np


def train_test_split(X, L, y, y_prime, f_prime, n_labeled, rand=0):
    index_pos = np.where(y > 0)[0]
    index_neg = np.where(y < 0)[0]

    n_labeled = int(n_labeled / 2)
    index_l = np.hstack((index_pos[:n_labeled],
                         index_neg[:n_labeled]))

    index_u = np.hstack((index_pos[n_labeled:],
                         index_neg[n_labeled:]))

    y_l = y[index_l]
    y_u = y[index_u]

    y_prime_l = y_prime[index_l]
    y_prime_u = y_prime[index_u]

    f_prime_l = f_prime[index_l]
    f_prime_u = f_prime[index_u]

    X_l = X[index_l, :]
    X_u = X[index_u, :]

    # Rearrange L
    index = np.hstack((index_l, index_u))
    L = L[index, :]
    L = L[:, index]

    # Split L too.

    return X_l, X_u, L, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u

def train_test_split_with_fixed_test(X, L, y, y_prime, f_prime, n_labeled, n_test, rand=0):
    index_pos = np.where(y > 0)[0]
    index_neg = np.where(y < 0)[0]

    n_total = y.shape[0]
    ind_test = n_total - n_test

    n_labeled = int(n_labeled / 2)
    index_l = np.hstack((index_pos[:n_labeled],
                         index_neg[:n_labeled]))

    index_u = np.hstack((index_pos[ind_test:],
                         index_neg[ind_test:]))

    y_l = y[index_l]
    y_u = y[index_u]

    y_prime_l = y_prime[index_l]
    y_prime_u = y_prime[index_u]

    f_prime_l = f_prime[index_l]
    f_prime_u = f_prime[index_u]

    X_l = X[index_l, :]
    X_u = X[index_u, :]

    # Rearrange L
    index = np.hstack((index_l, index_u))
    L = L[index, :]
    L = L[:, index]

    # Split L too.

    return X_l, X_u, L, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u