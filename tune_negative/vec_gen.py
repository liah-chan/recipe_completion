import numpy as np
from gensim.models import word2vec as w2v
from gensim import models
import multiprocessing

#generate vectors of ingredient
# for negative in range(5,16):
# 	for size in range(150,201,10): #(50,201,10)
sentences = w2v.LineSentence('../data/recipes_as_input.txt')
model = w2v.Word2Vec(sentences, sg=1,
	size=35, window=14, min_count=120,
	negative=1,sample=1e-3,
	workers=multiprocessing.cpu_count(),hs=0,
	iter=15)
model.save_word2vec_format('./gensim_model_1', binary=False)
#print 'negative samples:',negative,'vector size:',size