from sklearn.svm import LinearSVC, SVC

from classifier.binary_classifier import BinaryClassifier


class SvmBinaryClassifier(BinaryClassifier):
    def __init__(self):
        BinaryClassifier.__init__(self)

    def exec(self, X_l, X_u, y_l, y_u, rbf=False):

        if rbf:
            clf = SVC()
        else:
            clf = LinearSVC()

        clf.fit(X_l, y_l)

        y_pred = clf.predict(X_u)
        accuracy = self.accuracy(y_u, y_pred)
        return accuracy
