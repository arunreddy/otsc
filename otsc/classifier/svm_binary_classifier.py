from classifier.binary_classifier import BinaryClassifier
from sklearn.svm import LinearSVC


class SvmBinaryClassifier(BinaryClassifier):
    def __init__(self):
        BinaryClassifier.__init__(self)

    def exec(self, X, L, D, y, y_prime, f_prime, n_labeled=100):
        X_l, X_u, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u = self.train_test_split(X, L,
                                                                                               y, y_prime,
                                                                                               f_prime, n_labeled)

        clf = LinearSVC()
        clf.fit(X_l, y_l)

        y_pred = clf.predict(X_u)
        accuracy = self.accuracy(y_u, y_pred)
        print('> SVM accuracy %0.2f' % accuracy)
