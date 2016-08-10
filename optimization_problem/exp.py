from recipes import Recipes
from random import randint
from functools import reduce
import statistics
import pickle
import numpy as np
import logging
from random import sample

logging.basicConfig(filename='./logging.txt', filemode='w', level=logging.INFO)
logging.info('start')

def main():
	recs = Recipes()
	# print(len(recs.rank(40, 'butter')))
	index_file = './index.pickle'
	test_index_file = './test_index.pickle'

	# gen_test_index(recs, test_index_file)
	test_idxs = load_test_index(test_index_file)

	# gen_index(recs,index_file)
	rank_list = list()
	removed_idxs = load_index(index_file)

	#calculate ranks
	# for indice in range(1,52376):
	for indice in test_idxs:
		recipe_id,reclength,removed_ingr = recs.rand_remove(indice,removed_idxs[indice-1])
		# removed_idxs.append(rm_idx)
		# print('recipe id: '+str(recipe_id))
		logging.info('recipe id: ' + str(recipe_id) + ' == recipe length: ' + str(reclength) + \
			' == removed ingredient: '+ removed_ingr+' == index in recipe: '+str(removed_idxs[indice-1]) )
		# logging.info('recipe length: '+str(reclength))
		rank = len(recs.rank(recipe_id, removed_ingr))
		logging.info('prediction rank: '+str(rank))
		rank_list.append(rank)
		# break
	# print(rank_list)

	rank_file = './rank_list_context+cooccur.pickle'
	with open(rank_file,'wb') as handle:
		pickle.dump(rank_list, handle)
	handle.close()



	#measure the performance
	# rank_list = [1,5,6,2,3,7,9,10,4,8,11]
	# rank_list = [244,244,244]
	rank_array = np.array(rank_list)
	avg_rank = reduce(lambda x,y: x+y, rank_list)/len(rank_list)
	# median_rank = statistics.median(sorted(rank_list))
	median_rank = np.median(sorted(rank_list))
	n_10 = len(np.where(rank_array<=10)[0])
	n_1 = len(np.where(rank_array==1)[0])

	n_neg = 247
	auc = 1 - (rank_array-1)/n_neg
	mean_auc = np.mean(auc)

	print('using contex+cooccur, median rank: '+str(median_rank))
	print('using contex+cooccur, average rank: '+str(avg_rank))
	print('number of rank with in 10: '+str(n_10))
	print('correct prediction: '+str(n_1))
	print('mean auc: '+str(mean_auc))

def gen_test_index(recs,test_index_file):
	# index_list = list()
	# for i in range(0,5500):
	# 	r = randint(0,52375)
	# 	index_list.append(r)
	index_list = sample(range(1,52376), 5500)

	with open(test_index_file,'wb') as handle:
		pickle.dump(sorted( index_list ), handle)
	handle.close()
	# return index_list

def load_test_index(test_index_file):
	with open(test_index_file,'rb') as handle:
		test_idxs = pickle.load(handle)
	handle.close()
	return test_idxs

def gen_index(recs,index_file):
	#generating list of random indexes of ingredient in the recipe to be removed.
	removed_idxs =list()
	for indice in range(1,52376):
		removed_idxs.append(recs.rand_idx(indice))
	
	with open(index_file,'wb') as handle:
		pickle.dump(removed_idxs, handle)
	handle.close()

def load_index(index_file):
	with open(index_file,'rb') as handle:
		removed_idxs = pickle.load(handle)
	handle.close()
	return removed_idxs

if __name__=="__main__":
	main()