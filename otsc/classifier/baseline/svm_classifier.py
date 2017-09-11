from otsc.classifier.binary_classifier import BinaryClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import column_or_1d

class SVMClassifier(BinaryClassifier):

    def __init__(self, _run, n_labeled):
        self._run = _run
        self.n_labeled = n_labeled

    def __str__(self):
        return 'SVM with linear kernef_p_ll n_labeled[%d]' % (self.n_labeled)

    def exec(self, X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, random_state):
        data_set_name = self._run.config['dataset']['data_set_name']

        y = column_or_1d(y_l, warn=True)
        print(y.shape)
        clf = LinearSVC()
        clf.fit(X_l, y)

        y_pred = clf.predict(X_u)
        accuracy = self.accuracy(y_u, y_pred)

        m = y_l.shape[0]
        self._run.log_scalar(
            '%s.svm.acc____%0.2d' % (data_set_name, random_state),
            accuracy, step=m)
        print('SVM',accuracy)
        return accuracy

