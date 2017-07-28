"""
Script to covernt the large movie review dataset to numpy format.
"""
from glob import glob
import os
import pandas as pd
from sqlalchemy import create_engine

from .config import DB_ENGINE, DATA_DIR

def read_files(dir,name,label):
  reviews = []
  for file in glob(os.path.join(DATA_DIR,dir,'*.txt')):
      file_name = os.path.basename(file)
      review_txt = open(file,'r',encoding='utf8').readlines()[0]
      print('> %s: %d lines'%(file_name,len(review_txt)))
      reviews.append([name,review_txt,label])
  
  return reviews



def read_data():
  # Read positive reviews
  
  reviews = []
  reviews.extend(read_files('train/pos','train',1))
  reviews.extend(read_files('train/neg','train',-1))
  reviews.extend(read_files('test/pos','test',1))
  reviews.extend(read_files('test/neg','test',-1))
  
  df = pd.DataFrame(reviews,columns = ['name','review_txt','label'])
  df['stanford_label'] = 0.0
  df['stanford_confidence_scores'] = ""

  df['nltk_label'] = 0.0
  df['nltk_confidence_scores'] = 0.0

  print(df.head())
  
  psql = create_engine(DB_ENGINE)
  df.to_sql('acl_imdb',psql,if_exists = 'replace')
  


if __name__ == '__main__':
    read_data()
