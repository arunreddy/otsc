import numpy as np
from classifier.classifier_utils import train_test_split_with_fixed_test
from classifier.svm_binary_classifier import SvmBinaryClassifier
from classifier.fista_one_binary_classifier import FistaOneBinaryClassifier
from matplotlib import pyplot as plt


class Experiments(object):
    def __init__(self, X, L, D, y, y_prime, f_prime):
        self.X = X
        self.L = L
        self.D = D
        self.y = y
        self.y_p = y_prime
        self.f_p = f_prime

    def compare_algorithms(self, dataset):
        x = []
        results = []
        for n_labeled in range(100, 2001, 100):
            X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u = train_test_split_with_fixed_test(self.X,
                                                                                                 self.L,
                                                                                                 self.y,
                                                                                                 self.y_p,
                                                                                                 self.f_p,
                                                                                                 n_labeled,
                                                                                                 9000)
            clf = SvmBinaryClassifier()
            svm_accuracy = clf.exec(X_l, X_u, y_l, y_u)

            lambda_1 = 0.8
            lambda_2 = 0.8

            f_accs = []
            print('> %d' % n_labeled)
            for lambda_1 in np.arange(0.3, 1.0, 0.2):
                # for lambda_2 in np.arange(0.1, 1.0, 0.2):

                clf = FistaOneBinaryClassifier(self.X, self.L, self.D, self.y, self.y_p, self.f_p)
                f_val, l_val, x_del_f, s_acc, f_acc = clf.exec(X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, y_p_l, y_p_u,
                                                               lambda_1,
                                                               lambda_2, iterations=100)
                f_accs.append(f_acc)

            results.append([svm_accuracy, s_acc, np.mean(f_accs), np.max(f_accs), np.std(f_accs)])
            x.append(n_labeled)

        results = np.asarray(results)
        x = np.asarray(x)

        plt.errorbar(x, results[:, 0], yerr=results[:, 4] * 0.2, c='g', linestyle='--', marker='D', label='SVM',
                     barsabove=True,capsize=3.)
        plt.plot(x, results[:, 1], c='r', linestyle='--', marker='*', label='Stanford')
        plt.errorbar(x, results[:, 2], yerr=results[:, 4], c='k', linestyle='--', marker='o', label='OTSC-lasso',
                     barsabove=True,capsize=3.)
        plt.errorbar(x, results[:, 3], yerr=results[:, 4], c='b', linestyle='--', marker='P', label='OTSC-elasticnet',
                     barsabove=True,capsize=3.)
        plt.legend()
        plt.xlabel('# of labeled examples')
        plt.ylabel('Classificaton accuracy')
        plt.xticks(x, rotation='vertical')
        plt.grid(linestyle='dotted')

        if dataset == 'imdb':
            plt.title('IMDB Dataset')
        elif dataset == 'amazon_fine_foods':
            plt.title('Amazon Fine Food Reviews')
        elif dataset == 'amazon_binary':
            plt.title('Amazon Reviews')

        plt.tight_layout()
        plt.savefig('accuracy_%s.png' % dataset, dpi=600)
