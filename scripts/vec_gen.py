import numpy as np
from gensim.models import word2vec as w2v
from gensim import models
import multiprocessing

#generate vectors of ingredient
for negative in range(5,16):
	for size in range(150,201,10): #(50,201,10)
		sentences = w2v.LineSentence('../data/recipes_as_input.txt')
		model = w2v.Word2Vec(sentences, size=size, window=32, min_count=20,
						negative=negative,workers=multiprocessing.cpu_count())
		model.save_word2vec_format('../gensim_vec/model_'+str(size)+'_'+str(negative), binary=False)
		print 'negative samples:',negative,'vector size:',size