"""
Read the data from database and generate tf-idf/word2vec features.
"""
from otsc.config import *
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csgraph
import numpy as np
import joblib


def tf_idf(D):
  vectorizer = TfidfVectorizer(ngram_range = [1, 3], min_df = 5)
  return vectorizer.fit_transform(D)

def tf_bin(D):
  vectorizer = CountVectorizer(binary = True, ngram_range = [1,3], min_df = 5)
  return vectorizer.fit_transform(D)

def word2vec(D):
  pass



def genereate_features(df, t = 'tf-idf'):
  
  if t == 'tf-idf':
    X = tf_idf(df['review_txt'])
  
  elif t == 'binary':
    X = tf_bin(df['review_txt'])
  
  elif   t == 'word2vec':
    X = word2vec(df['review_txt'])


  return X


def graph_laplacian(X, t_sim='cosine', normed=True):
  
  # Compute the similarity matrix A
  if t_sim == 'cosine':
    A = cosine_similarity(X)
  
  elif t_sim == 'sxd':
    pass
  
  
  # Laplacian.
  L, D = csgraph.laplacian(A,normed = normed, return_diag = True)
  
  return L, D

def load_data(n):
  df_pos = pd.read_sql('SELECT * FROM acl_imdb WHERE label=1 LIMIT %d'%n,
                       con = create_engine(DB_ENGINE), parse_dates = True)
  df_neg = pd.read_sql('SELECT * FROM acl_imdb WHERE label=-1 LIMIT %d'%n,
                       con = create_engine(DB_ENGINE), parse_dates = True)

  df_pos['f_prime'] = df_pos['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
  df_neg['f_prime'] = df_neg['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
  return df_pos, df_neg

def generate_features(df_pos, df_neg, n, n_labeled):
  df_pos_l = df_pos[:n_labeled]
  df_neg_l = df_neg[:n_labeled]
  df_pos_u = df_pos[n_labeled:]
  df_neg_u = df_neg[n_labeled:]
  
  # Combine all the data.
  df_l = df_pos_l.append(df_neg_l)
  df_u = df_pos_u.append(df_neg_u)
  df = df_l.append(df_u)
  
  # Laplacian matrix.
  X = genereate_features(df)
  L, D = graph_laplacian(X)
  
  # Known labels.
  y_l = [1] * df_pos_l.shape[0] + [-1] * df_neg_l.shape[0]
  y_u = [1] * df_pos_u.shape[0] + [-1] * df_neg_u.shape[0]

  # OTSC labels.
  y_prime_l = np.hstack((df_pos_l['stanford_label'].values, df_neg_l['stanford_label'].values))
  y_prime_u = np.hstack((df_neg_u['stanford_label'].values, df_neg_u['stanford_label'].values))
  
  # OTSC confidence scores.
  f_prime_l = np.hstack((df_pos_l['f_prime'].values, df_neg_l['f_prime'].values))
  f_prime_u = np.hstack((df_pos_u['f_prime'].values, df_neg_u['f_prime'].values))
  
  return L, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u

if __name__ == '__main__':
  
  # Parameters.
  n = 2000
  n_labeled = 500
  
  df_pos, df_neg = load_data(n)
  L, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u = generate_features(df_pos, df_neg, n, n_labeled)

  print(len(y_l))
  
  y_prime = np.asarray(y_prime_l)

  y_prime = y_prime-2
  
  count = 0
  for i in range(len(y_l)):
    if (y_l[i]*y_prime[i] > 0):
      count+=1
  
  print(count/len(y_l))

  
  joblib.dump([L, D, y_l, y_u, y_prime_l, y_prime_u, f_prime_l, f_prime_u],'features.dat',compress = 5)