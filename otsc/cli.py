import argparse
import logging
import hashlib

from executor import Executor


def update_configuration(args):
  return args


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='OTSC - Off-the-Shelf Classifier')
  parser.add_argument('--dataset', choices=['imdb', 'amazon_fine_foods', 'amazon_binary'], default='imdb')
  parser.add_argument('--feat', choices=['bin', 'tf-idf', 'word2vec'], default='tf-idf')
  parser.add_argument('--sim', choices=['cosine', 'rbf'], default='cosine')
  parser.add_argument('--n-iterations', type=int, default=10)
  parser.add_argument('--n-total', type=int, default=1000)
  parser.add_argument('--n-labeled', type=int, default=400)
  parser.add_argument('--weighted', action='store_true', default=True)
  parser.add_argument('--debug', action='store_true', default=True)

  args = parser.parse_args()

  # Load the configuration from file.
  args = update_configuration(args)
  args.data_home = '/home/arun/code/github/otsc/data/data'
  args.results_dir = '/home/arun/code/github/otsc/data/results/methods'
  args.args_hash = hashlib.md5(str(args).encode('utf-8')).hexdigest()

  # set up logging.
  log_format = '%(asctime)s - %(name)-12s - %(levelname)-8s: %(message)s'
  log_level = logging.DEBUG if args.debug else logging.INFO
  logging.basicConfig(level=log_level, format=log_format)

  # Execute the code.
  executor = Executor(args)
  executor.execute()