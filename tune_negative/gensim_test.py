import numpy as np
from gensim.models import Word2Vec as w2v
from gensim import models
import multiprocessing

#model = w2v.load('./gensim_model_1')
model = w2v.load_word2vec_format('./gensim_model_1', binary=False)
print model.most_similar('gelatin')