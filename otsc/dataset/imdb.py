"""
Script to covernt the large movie review dataset to numpy format.
"""
from glob import glob
import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from config import DB_ENGINE, DATA_DIR


class Imdb(object):
    def __init__(self):
        pass

    def read_files(self, dir, name, label):
        reviews = []
        for file in glob(os.path.join(DATA_DIR, dir, '*.txt')):
            file_name = os.path.basename(file)
            review_txt = open(file, 'r', encoding='utf8').readlines()[0]
            print('> %s: %d lines' % (file_name, len(review_txt)))
            reviews.append([name, review_txt, label])

        return reviews

    def read_data(self):
        # Read positive reviews

        reviews = []
        reviews.extend(self.read_files('train/pos', 'train', 1))
        reviews.extend(self.read_files('train/neg', 'train', -1))
        reviews.extend(self.read_files('test/pos', 'test', 1))
        reviews.extend(self.read_files('test/neg', 'test', -1))

        df = pd.DataFrame(reviews, columns=['name', 'review_txt', 'label'])
        df['stanford_label'] = 0.0
        df['stanford_confidence_scores'] = ""

        df['nltk_label'] = 0.0
        df['nltk_confidence_scores'] = 0.0

        print(df.head())

        psql = create_engine(DB_ENGINE)

        df.to_sql('acl_imdb', psql, if_exists='replace')

    def load_data(self, n):
        df_pos = pd.read_sql('SELECT * FROM acl_imdb WHERE label=1 ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)
        df_neg = pd.read_sql('SELECT * FROM acl_imdb WHERE label=-1 ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)

        df_pos['f_prime'] = df_pos['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
        df_neg['f_prime'] = df_neg['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
        return df_pos, df_neg
