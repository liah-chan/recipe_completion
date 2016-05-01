import numpy as np
from random import randint
from scipy.spatial.distance import cosine
import sys

model_file = '../word2vec_vec_cbow_ns_16/model_200_5'
vec_data = np.genfromtxt(model_file,delimiter=' ',
	dtype=None, #(199,110)
	usecols=(_ for _ in range (1,201)),
	autostrip=True,skip_header=2)
ing_name = np.genfromtxt(model_file,delimiter=' ', #0-198
	dtype=None, #(199,)
	usecols=(0),
	autostrip=True,skip_header=2)
vec_size = len(vec_data[1])
total = 56498
good = 0
predictions = list()
cnt = 0
with open('../data/recipes_as_input.txt','r') as f:
	for line in f.readlines():
		vec_data_ = np.copy(vec_data)
		ingredients = line.strip('\n').split(' ');
		if all(ingredient not in ing_name for ingredient in ingredients):
			total -= 1
			continue
		i = 0
		r = randint(0,len(ingredients)-1)
		#print r
		removed = ingredients[r]
		while (removed not in ing_name):
			r = randint(0,len(ingredients)-1)
			removed = ingredients[r]
			#print removed
		removed_index = list(ing_name).index(removed)
		#print 'removed:',removed,'index:',removed_index		 
		del ingredients[r]
		#print ingredients
		input_vec = np.zeros((len(ingredients),vec_size))
		indexs = list()
		for ingredient in ingredients:
			if ingredient in ing_name:
				index = list(ing_name).index(ingredient)
				input_vec[i] = vec_data[index,:]
				indexs.append(index)
				i += 1
		vec_data_[indexs,:] = 0

		output = np.zeros(len(vec_data_))
		for vec in range(len(vec_data_)): #1-195
			if any(vec_data_[vec,i] != 0 for i in range(0,vec_size) ):
				profit = 0
				for m in range(len(input_vec)): #length of the input recipe
					if any(input_vec[m,i] != 0 for i in range(0,vec_size)):
						profit += 1-cosine(vec_data_[vec], input_vec[m])
					else:
						continue
				output[vec] = profit
				#print output.shape
				best = np.argsort(output,axis=0)[::-1]
				#print best
				rank = list(best).index(removed_index)
			else:
				continue
		
		if rank <= 10:
			good += 1
		predictions.append(rank)
		#print 'prediction:',list(best).index(removed_index)
		if cnt%1000 == 0:
			print str(cnt)+'th recipe'
			print 'total:',total,'======within 10:',good
		cnt += 1

f.close()
# pred = np.array(predictions)
# res = pred.where(pred <= 10)

# print 'total within 10:',len(res)
# print 'total :',total


