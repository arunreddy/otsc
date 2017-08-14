from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec
import nltk
import numpy as np


class Doc2VecUtil(object):
    def __init__(self, docs):
        self.docs = docs

    def fit(self):
        tagged_doc_list = []
        tagged_doc = namedtuple('TaggedDocument', 'words tags')

        for idx, doc in enumerate(self.docs):
            tokens = nltk.word_tokenize(doc)
            tokens_filtered = [t for t in tokens if t.isalpha()]

            tag = [idx]
            tagged_doc_list.append(tagged_doc(tokens_filtered, tag))

        model = Doc2Vec(tagged_doc_list, size=100, window=300, min_count=1, workers=4)
        return np.asarray(model.docvecs)

# Testing..
# if __name__ == '__main__':
#     import lipsum
#
#     docs = []
#     for i in range(10):
#         docs.append(lipsum.generate_paragraphs(2))
#
#     obj = Doc2VecUtil(docs)
#     obj.fit()
