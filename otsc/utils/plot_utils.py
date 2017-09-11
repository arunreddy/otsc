import matplotlib.pyplot as plt
import numpy as np
from bson.objectid import ObjectId
from pymongo import MongoClient
import logging

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
