from classifier.binary_classifier import BinaryClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class SvmBinaryClassifier(BinaryClassifier):
    def __init__(self):
        BinaryClassifier.__init__(self)

    def exec(self, X_l, X_u, y_l, y_u):

        clf = LinearSVC()
        clf.fit(X_l, y_l)

        y_pred = clf.predict(X_u)
        accuracy = self.accuracy(y_u, y_pred)
        print(accuracy)
        return accuracy
