from config import DB_ENGINE, DATA_DIR
import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

class AmazonFineFoodReviews(object):

    def __init__(self):
        pass

    def import_data_to_db(self):
        df = pd.read_csv(os.path.join(DATA_DIR, 'amazon-fine-food-reviews/Reviews.csv'))
        psql = create_engine(DB_ENGINE)
        df.to_sql('amazon_fine_food_reviews', psql, if_exists='replace')

    def load_data(self, n):
        df_pos = pd.read_sql('SELECT * FROM amazon_fine_food_reviews WHERE "Score"=5 and stanford_label is not null ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)
        df_neg = pd.read_sql('SELECT * FROM amazon_fine_food_reviews WHERE "Score"=1 and stanford_label is not null ORDER BY index ASC LIMIT %d ' % n,
                             con=create_engine(DB_ENGINE), parse_dates=True)

        df_pos['f_prime'] = df_pos['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
        df_neg['f_prime'] = df_neg['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))

        df_pos['label'] = 1
        df_neg['label'] = -1

        df_pos['review_txt'] = df_pos['Text']
        del df_pos['Text']

        df_neg['review_txt'] = df_neg['Text']
        del df_neg['Text']

        print(df_pos.head())
        print(df_neg.head())

        return df_pos, df_neg
