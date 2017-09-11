import matplotlib.pyplot as plt
import numpy as np
from bson.objectid import ObjectId
from pymongo import MongoClient
import logging

def plot(id, lbl, show_stanford=True):
    client = MongoClient()
    db = client.sacred

    svm = []
    fista_lasso = []
    fista_lasso_boost = []
    fista_lasso_cen = []
    fista_lasso_boost_cen = []
    stanford = []
    stanford_cen = []
    steps = None

    for doc in db.runs.find({'_id': id}):
        for metric in doc['info']['metrics']:
            metrics = db.metrics.find({'_id': ObjectId(metric['id'])})
            for m in metrics:
                steps = m['steps']
                name = m['name']

                if 'svm.acc' in name:
                    svm.append(m['values'])
                elif 'fista_lasso.s_acc_cen' in name:
                    stanford_cen.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.s_acc' in name:
                    stanford.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.f_acc_boost_cen' in name:
                    fista_lasso_boost_cen.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.f_acc_boost' in name:
                    fista_lasso_boost.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.f_acc_cen' in name:
                    fista_lasso_cen.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.f_acc' in name:
                    fista_lasso.append([(x + y) / 8000 for [x, y] in m['values']])

    fista_lasso = np.asarray(fista_lasso, dtype=float)
    fista_lasso_boost = np.asarray(fista_lasso_boost, dtype=float)
    fista_lasso_cen = np.asarray(fista_lasso_cen, dtype=float)
    fista_lasso_boost_cen = np.asarray(fista_lasso_boost_cen, dtype=float)

    # plt.errorbar(steps, np.mean(np.asarray(svm), axis=0), yerr=np.std(np.asarray(svm), axis=0), label='svm%s'%lbl, linestyle='--',
    #              marker='*', barsabove=True, capsize=3.)
    plt.errorbar(steps, np.mean(fista_lasso, axis=0), yerr=np.std(fista_lasso, axis=0),
                 label='otsc%s' % lbl, linestyle='--', marker='*', barsabove=True, capsize=3.)
    plt.errorbar(steps, np.mean(fista_lasso_boost, axis=0), yerr=np.std(fista_lasso_boost, axis=0),
                 label='otsc_boost%s' % lbl, linestyle='--', marker='*', barsabove=True, capsize=3.)
    plt.errorbar(steps, np.mean(fista_lasso_cen, axis=0), yerr=np.std(fista_lasso_cen, axis=0),
                 label='otsc_recal%s' % lbl, linestyle='--', marker='*', barsabove=True, capsize=3.)
    plt.errorbar(steps, np.mean(fista_lasso_boost_cen, axis=0), yerr=np.std(fista_lasso_boost_cen, axis=0),
                 label='otsc_recal_boost%s' % lbl, linestyle='--', marker='*', barsabove=True, capsize=3.)

    if show_stanford:
        plt.errorbar(steps, np.mean(np.asarray(stanford), axis=0), yerr=np.std(np.asarray(stanford), axis=0),
                     label='stanford',
                     linestyle='--', marker='*', barsabove=True, capsize=3.)
        plt.errorbar(steps, np.mean(np.asarray(stanford_cen), axis=0), yerr=np.std(np.asarray(stanford_cen), axis=0),
                     label='stanford_recal', linestyle='--', marker='*', barsabove=True, capsize=3.)

    # plt.xticks(steps, rotation='vertical')
    plt.xticks([x for x in steps if x % 4 == 0], rotation='vertical')
    plt.ylabel('Classificaton accuracy')
    plt.xlabel('# of labeled examples')


def plot_pos_neg(id, lbl, show_stanford=True):
    client = MongoClient()
    db = client.sacred

    svm = []
    fista_lasso_pos = []
    fista_lasso_neg = []
    fista_lasso_boost_pos = []
    fista_lasso_boost_neg = []
    fista_lasso_cen_pos = []
    fista_lasso_cen_neg = []
    stanford = []
    stanford_cen = []
    steps = None

    for doc in db.runs.find({'_id': id}):
        for metric in doc['info']['metrics']:
            metrics = db.metrics.find({'_id': ObjectId(metric['id'])})
            for m in metrics:
                steps = m['steps']
                name = m['name']

                if 'svm.acc' in name:
                    svm.append(m['values'])
                elif 'fista_lasso.s_acc_cen' in name:
                    stanford_cen.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.s_acc' in name:
                    stanford.append([(x + y) / 8000 for [x, y] in m['values']])
                elif 'fista_lasso.f_acc_cen' in name:
                    fista_lasso_cen_pos.append([x for [x, y] in m['values']])
                    fista_lasso_cen_neg.append([y for [x, y] in m['values']])
                elif 'fista_lasso.f_acc' in name:
                    fista_lasso_pos.append([x for [x, y] in m['values']])
                    fista_lasso_neg.append([y for [x, y] in m['values']])
                elif 'fista_lasso.f_acc_boost' in name:
                    fista_lasso_boost_pos.append([x for [x, y] in m['values']])
                    fista_lasso_boost_neg.append([y for [x, y] in m['values']])

    # plt.errorbar(steps, np.mean(np.asarray(fista_lasso_pos), axis=0), yerr=np.std(np.asarray(fista_lasso_pos), axis=0),
    #              label='otsc_pos%s'%lbl, linestyle='--', marker='*', barsabove=True, capsize=3.,c='red')
    # plt.errorbar(steps, np.mean(np.asarray(fista_lasso_neg), axis=0), yerr=np.std(np.asarray(fista_lasso_neg), axis=0),
    #              label='otsc_neg%s'%lbl, linestyle='--', marker='*', barsabove=True, capsize=3.,c='red')
    plt.errorbar(steps, np.mean(np.asarray(fista_lasso_cen_pos), axis=0),
                 yerr=np.std(np.asarray(fista_lasso_cen_pos), axis=0),
                 label='otsc_pos%s' % lbl, linestyle='--', marker='*', barsabove=True, capsize=3., c='blue')
    plt.errorbar(steps, np.mean(np.asarray(fista_lasso_cen_neg), axis=0),
                 yerr=np.std(np.asarray(fista_lasso_cen_neg), axis=0),
                 label='otsc_neg%s' % lbl, linestyle='--', marker='D', barsabove=True, capsize=3., c='blue')

    # plt.xticks(steps, rotation='vertical')
    plt.xticks([x for x in steps if x % 4 == 0], rotation='vertical')
    plt.ylabel('# of examples in the test set')
    plt.xlabel('# of labeled examples')


class PlotUtils(object):

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

    def plot_and_save_results(self, run_number):
        self.logger.info('Plotting the results for the run {}',run_number)

        client = MongoClient()
        db = client.sacred

        items = []
        for doc in db.runs.find({'_id': id}):
            for metric in doc['info']['metrics']:
                metrics = db.metrics.find({'_id': ObjectId(metric['id'])})
                for m in metrics:
                    steps = m['steps']
                    name = m['name']
                    if '____' in name:
                        items.append(name.split('____')[0])

        items = np.unique(items)

        item_dict = {}
        for item in items:
            item_dict[item] = []

        for doc in db.runs.find({'_id': id}):
            for metric in doc['info']['metrics']:
                metrics = db.metrics.find({'_id': ObjectId(metric['id'])})
                for m in metrics:
                    steps = m['steps']
                    name = m['name']
                    if '____' in name:
                        key = name.split('____')[0]
                        #
                        if type(m['values'][0]) is list:
                            item_dict[key].append([(x + y) / 8000 for [x, y] in m['values']])
                        else:
                            item_dict[key].append(m['values'])

        labels = {}

        labels['svm.acc'] = 'SVM'
        labels['xgb.acc'] = 'Gradient Boost'
        labels['fista_lasso_svr_.s_acc'] = 'Stanford'
        labels['fista_lasso_svr_.s_acc_cen'] = 'Stanford Calibrated'

        labels['fista_lasso_svr_.f_acc'] = 'AOTSC'
        labels['fista_lasso_svr_.f_acc_cen'] = 'AOTSC Calibrated'

        labels['fista_lasso_boost_ xgb_.f_acc_boost'] = 'AOTSC gradient boost'
        labels['fista_lasso_boost_svr_.f_acc_boost'] = 'AOTSC SVR boost'

        labels['fista_lasso_svr_adv.f_acc_boost'] = 'AOTSC SVR boost + SVM'
        labels['fista_lasso_svr_adv.s_acc_cen'] = 'Stanford Calibrated + SVM'
        labels['fista_lasso_svr_adv.f_acc_cen'] = 'AOTSC Calibrated + SVM'

        for item in items:
            v = np.asarray(item_dict[item])
            if len(steps) == v.shape[1]:

                items = item.split('.', 1)
                dataset = items[0]
                algo = items[1].strip('_')
                print(algo)

                if algo in labels:
                    plt.errorbar(steps, np.mean(v, axis=0),
                                 yerr=np.std(v, axis=0),
                                 label=labels[algo], linestyle='--', marker='*', barsabove=True, capsize=3.)

        #
    #
    # plt.figure(figsize=(8.5, 6))
    # # for id in [154,152]:
    # #     plot(id,'')
    # generic_plot(346)
    # # plot(157, '_rbf',show_stanford=False)
    # # plot_pos_neg(274, '_cosine')
    # # plot(151, '_rbf',show_stanford=False)
    #
    #
    #
    # plt.grid(linestyle='dotted')
    # plt.ylabel('Classification accuracy on the test set')
    # plt.xlabel('# of labeled examples')
    # plt.legend()
    # plt.tight_layout(pad=2.0)
    # plt.title('IMDB reviews dataset')
    # #
    # plt.savefig('imdb_SVR_SVM_100.png', dpi=600)
    # # plt.show()
    # #
