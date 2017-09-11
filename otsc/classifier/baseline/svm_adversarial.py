import numpy as np
from sklearn.svm import LinearSVC

from otsc.classifier.binary_classifier import BinaryClassifier


class SVMAdversarial(BinaryClassifier):
    def __init__(self, _run, n_labeled):
        self._run = _run
        self.n_labeled = n_labeled

    def __str__(self):
        return 'SVM with linear kernef_p_ll n_labeled[%d]' % (self.n_labeled)

    def balanced_sample_maker(self, X, y, sample_size, random_seed=None):
        """ return a balanced data set by sampling all classes with sample_size
            current version is developed on assumption that the positive
            class is the minority.

        Parameters:
        ===========
        X: {numpy.ndarrray}
        y: {numpy.ndarray}
        """
        uniq_levels = np.unique(y)
        uniq_counts = {level: sum(y == level) for level in uniq_levels}

        if not random_seed is None:
            np.random.seed(random_seed)

        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(y) if val == level]
            groupby_levels[level] = obs_idx
        # oversampling on observations of each label
        balanced_copy_idx = []
        for gb_level, gb_idx in groupby_levels.iteritems():
            over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
            balanced_copy_idx += over_sample_idx
        np.random.shuffle(balanced_copy_idx)

        return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)

    def exec(self, X_l, X_u, L, y_l, y_u, y_p_l, y_p_u, f_p_l, f_p_u, random_state):
        data_set_name = self._run.config['dataset']['data_set_name']

        # Scale the labels to the range [-1,1]
        _y_p_l = np.copy(y_p_l)
        _y_p_l[_y_p_l < 0] = -1
        _y_p_l[_y_p_l > 0] = 1

        _y_p_u = np.copy(y_p_u)
        _y_p_u[_y_p_u < 0] = -1
        _y_p_u[_y_p_u > 0] = 1

        _yl = []
        for i in range(y_l.shape[0]):
            if y_l[i] == _y_p_l[i]:
                _yl.append(1)
            else:
                _yl.append(-1)

        _yu = []
        for i in range(y_u.shape[0]):
            if y_u[i] == _y_p_u[i]:
                _yu.append(1)
            else:
                _yu.append(-1)

        # Print the statistics.
        print(np.unique(_yl, return_counts=True))
        print(np.unique(_yu, return_counts=True))

        _yl = np.asarray(_yl)
        _yu = np.asarray(_yu)


        # Resample the training data.
        pos_idx = np.where(_yl>0)[0]
        neg_idx = np.where(_yl<0)[0]

        np.random.shuffle(pos_idx)
        idx = np.hstack((pos_idx[:neg_idx.shape[0]],neg_idx))
        print(idx.shape)

        clf = LinearSVC()
        clf.fit(X_l[idx,:], _yl[idx])
        ayl = clf.predict(X_l)
        ayu = clf.predict(X_u)
        ayl = np.reshape(ayl, newshape=(ayl.shape[0], 1))
        ayu = np.reshape(ayu, newshape=(ayu.shape[0],1))
        #
        # accuracy = self.accuracy(_yu, y_pred)
        # print('SVM Accuracy', accuracy)


        return ayl, ayu

        # yx = np.multiply(y_pred, _yu)

        # from costcla.models import CostSensitiveLogisticRegression
        # clf = CostSensitiveLogisticRegression()



        #
        #
        # print(y_pred.shape, _y_p_u.shape)
        #
        # print(yx.shape)
        #
        # cnt = 0
        # for i in range(y_u.shape[0]):
        #     if y_u[i] == _y_p_u[i]:
        #         cnt += 1.0
        # print(cnt / y_u.shape[0])
        #
        # cnt = 0
        # for i in range(y_u.shape[0]):
        #     if y_u[i] == yx[i]:
        #         cnt += 1.0
        # print(cnt / y_u.shape[0])
        #



        # import xgboost as xgb
        # self.compute_accuracy(xgb.XGBClassifier(max_depth=2), X_l, _yl, X_u, _yu)





        # #
        # m = y_l.shape[0]
        # self._run.log_scalar(
        #     '%s.svm.acc____%0.2d' % (data_set_name, random_state),
        #     accuracy, step=m)
        # return accuracy
