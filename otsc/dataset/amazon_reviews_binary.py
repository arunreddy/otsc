from config import DB_ENGINE, DATA_DIR
import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

class AmazonReviewsBinary(object):
    def __init__(self):
        pass

    def import_data_to_db(self):
        pos_text = None
        neg_text = None
        with open(os.path.join(DATA_DIR, 'amazonreviews'
                                         '/train_label1.txt')) as f:
            pos_text = f.readlines()

        with open(os.path.join(DATA_DIR, 'amazonreviews'
                                         '/train_label2.txt')) as f:
            neg_text = f.readlines()

        df_pos = pd.DataFrame(pos_text, columns=['text'])
        df_pos['label'] = 1

        df_neg = pd.DataFrame(neg_text, columns=['text'])
        df_neg['label'] = -1

        df = df_pos.append(df_neg)

        psql = create_engine(DB_ENGINE)
        df.to_sql('amazon_reviews_tenk', psql, if_exists='replace')


    def load_data(self, n):
        df_pos = pd.read_sql('SELECT * FROM amazon_reviews_tenk WHERE label=-1 ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)
        df_neg = pd.read_sql('SELECT * FROM amazon_reviews_tenk WHERE label=1 ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)

        df_pos['f_prime'] = df_pos['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
        df_neg['f_prime'] = df_neg['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))

        df_pos['review_txt'] = df_pos['text']
        del df_pos['text']

        df_neg['review_txt'] = df_neg['text']
        del df_neg['text']

        return df_pos, df_neg

if __name__ == '__main__':
    amazon_fine_food_reviews = AmazonReviewsBinary()
    amazon_fine_food_reviews.import_data_to_db()
