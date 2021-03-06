"""
Read the data from database and generate tf-idf/word2vec features.
"""
from config import *
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy.sparse import csgraph
import numpy as np
import joblib
from scipy import sparse
from feat.doc2vec_util import Doc2VecUtil

class FeaturesGenerator(object):
    def __init__(self):
        pass

    def tf_idf(self, D):
        vectorizer = TfidfVectorizer(ngram_range=[1, 4],min_df=20, max_df=3000)
        return vectorizer.fit_transform(D)

    def tf_bin(self, D):
        vectorizer = CountVectorizer(binary=True, ngram_range=[1, 4], min_df=20, max_df=3000)
        return vectorizer.fit_transform(D)

    def genereate_features(self, df, t='tf-idf'):

        if t == 'tf-idf':
            X = self.tf_idf(df['review_txt'])

        elif t == 'bin':
            X = self.tf_bin(df['review_txt'])

        elif t == 'word2vec':
            obj = Doc2VecUtil(df['review_txt'])
            X = obj.fit()

        return X

    def graph_laplacian(self, X, t_sim='cosine', normed=True):

        # Compute the similarity matrix A
        if t_sim == 'cosine':
            A = cosine_similarity(X)
            A[A < .1] = 0.

        elif t_sim == 'rbf':
            A = rbf_kernel(X, gamma=1.)
            print(np.max(A), np.mean(A), np.min(A))
            A[A < .2] = 0.


        # Laplacian.
        L, D = csgraph.laplacian(A, normed=normed, return_diag=True)
        L = sparse.csc_matrix(L)
        L.eliminate_zeros()
        print('Nonzeros:',L.nnz)
        print('Sparsity:',(L.nnz*100.)/(L.shape[0]*L.shape[1]))

        return L, D

    def generate_features(self, df_pos, df_neg, n, feat_type, sim):

        print("> Generating features..")
        print("\t Positive: %d" % (df_pos.shape[0]))
        print("\t Negative: %d" % (df_neg.shape[0]))
        # Combine all the data.
        df = df_pos.copy().append(df_neg)

        # Laplacian matrix.
        X = self.genereate_features(df,feat_type)
        L, D = self.graph_laplacian(X,t_sim=sim)

        # Known labels.
        y = np.asarray([1] * df_pos.shape[0] + [-1] * df_neg.shape[0])

        # OTSC labels.
        y_prime = np.hstack((df_pos['stanford_label'].values, df_neg['stanford_label'].values))
        y_prime = y_prime - 2

        # OTSC confidence scores.
        f_prime = np.hstack((df_pos['f_prime'].values, df_neg['f_prime'].values))

        # Reshape the arrays.
        f_prime = f_prime.reshape(f_prime.shape[0], 1)
        y_prime = y_prime.reshape(y_prime.shape[0], 1)
        y = y.reshape(y.shape[0], 1)

        print(L.shape)

        return X, L, D, y, y_prime, f_prime
