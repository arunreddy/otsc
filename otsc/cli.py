import argparse
from dataset import Imdb, AmazonReviewsBinary, AmazonFineFoodReviews
from features import FeaturesGenerator
import joblib
import numpy as np
from classifier.svm_binary_classifier import SvmBinaryClassifier
from classifier.fista_one_binary_classifier import FistaOneBinaryClassifier
from experiments import Experiments

if __name__ == '__main__':
    print("##################################################")
    print("                      OTSC")
    print("##################################################")

    parser = argparse.ArgumentParser(description='OTSC - Off-the-Shelf Classifier')
    parser.add_argument('--gen-data', action='store_true')
    parser.add_argument('--dataset', choices=['imdb', 'amazon_fine_foods', 'amazon_binary'])
    parser.add_argument('--feat', choices=['bin', 'tfidf', 'word2vec'], default='bin')
    parser.add_argument('--analyze-data', action='store_true')
    parser.add_argument('--n-iterations', type=int, default=10)
    parser.add_argument('--n-total', type=int, default=1000)
    parser.add_argument('--n-labeled', type=int, default=400)
    parser.add_argument('--run', choices=['all', 'otsc_one', 'otsc_two'])

    args = parser.parse_args()
    n_total = args.n_total
    n_labeled = args.n_labeled
    out_file_name = "/tmp/%s_%d.dat" % (args.dataset, n_total)

    if args.gen_data:

        dataset = None
        # Run the dataset.
        if args.dataset == 'imdb':
            dataset = Imdb()
        elif args.dataset == 'amazon_fine_foods':
            dataset = AmazonReviewsBinary()
        elif args.dataset == 'amazon_binary':
            dataset = AmazonFineFoodReviews()
        else:
            print('> Specify the dataset.')

        # Generate the features.
        df_pos, df_neg = dataset.load_data(n_total)

        feat_generator = FeaturesGenerator()
        X, L, D, y, y_prime, f_prime = feat_generator.generate_features(df_pos, df_neg, n_total)

        print(L)

        print(np.nonzero(L))

        print('> Dumping the data into %s' % out_file_name)
        joblib.dump([X, L, D, y, y_prime, f_prime], out_file_name, compress=5)

        if args.analyze_data:

            print('> Stanford Accuracy.')

            count = 0
            for i in range(len(y)):
                if y[i] * y_prime[i] > 0:
                    count += 1
            print('\t accuracy %0.2f' % (count / len(y)))
            print('\t unique true %s' % (np.unique(y)))
            print('\t unique stanford %s' % (np.unique(y_prime)))


    if args.run == 'all':
        X, L, D, y, y_prime, f_prime = joblib.load(out_file_name)
        experiments = Experiments(X, L, D, y, y_prime, f_prime)
        experiments.compare_algorithms(args.dataset)

    elif args.run == 'otsc_one':
        X, L, D, y, y_prime, f_prime = joblib.load(out_file_name)

        # clf = SvmBinaryClassifier()
        # clf.exec(X, L, D, y, y_prime, f_prime, n_labeled)

        # clf = FistaOneBinaryClassifier(X, L, D, y, y_prime, f_prime)
        # clf.analyze_data(n_labeled, args.n_iterations)


    else:
        print('> Please provide the args run type')
