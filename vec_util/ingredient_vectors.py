#system module
import numpy as np
import os
import sys
from collections import defaultdict
from gensim.models import word2vec as w2v
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import dice
from scipy.spatial.distance import pdist,squareform
#local module
sys.path.append('../vec_util/')
from matrix import get_common_ingr_matrix
from plotting import plot_similarity_by_category,plot_similarity_distribution
import pickle
import Queue
from threading import Thread

ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))
recipe_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipes_major.txt'))
reci_ingr_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/reci_ingr_matrix.txt'))

similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/similarity_matrix.txt'))
normalized_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/normalized_similarity_matrix.txt'))
cosine_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/cosine_similarity_matrix.txt'))
angular_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/angular_similarity_matrix.txt'))

ingredient_dict_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredient_dict.pickle'))
recipe_dict_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipe_dict.pickle'))

best_vector_file = os.path.realpath(os.path.join(os.getcwd(),'../tune_size/model_10'))


if not os.path.isfile(reci_ingr_matrix_file):
	matrix_gen(ingredient_file, recipe_file)

# if not os.path.isfile(cosine_similarity_matrix_file):
# 	gensim_matrix_gen()

#read recipe ingredient matrix (52375, 248) from file
reci_ingr_matrix = np.genfromtxt(reci_ingr_matrix_file,
		delimiter=' ',
		dtype=int,
		#usecols=(0),
		autostrip=True,skip_header=0)
# print reci_ingr_matrix.shape #(52375, 248)

# common_ingr_matrix = get_common_ingr_matrix(ingredient_file, recipe_file)
# print common_ingr_matrix.shape

#store 248 major ingredients and their food category into arrays 
ingr_all,cate_all = np.genfromtxt(ingredient_file, #248 major ingredients
		delimiter=',',
		dtype=None,
		usecols=(0,1),
		autostrip=True,
		skip_header=0,
		unpack=True)

#read similarity matrix (248x248) from file
similarity_matrix = np.genfromtxt(similarity_matrix_file,
		delimiter=' ',
		dtype=float,
		#usecols=(0),
		autostrip=True,skip_header=0)

def main():
	save_angular_similarity_matrix()


# def recipe_level_similarity(ingr_index_i,ingr_index_j):
# 	"""
# 	    compute the raw similarity between ingredients according to the recipes that contain them,
# 	    by the formular
# 	    .. math::
# 	       sim(i,j) = \frac{1}{n_I}\frac{1}{n_J}\sum_{I_{R_i} \in I}\sum_{ I_{R_j} \in J} <I_{R_i},I_{R_j}>.

# 	    Parameters
# 	    ----------
# 	    ingr_index_i : int
# 	        the index of the first ingredient
# 	    ingr_index_j : int
# 	        the index of the second ingredient

# 	    Returns
# 	    -------
# 	    r_similarity : double
# 	        The raw similarity score between two ingredients.
#     """

# 	i_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_i] == 1)
# 	j_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_j] == 1)
# 	# print i_recipes_indexes,j_recipes_indexes
# 	sim_sum = 0
# 	for i in i_recipes_indexes[0]:
# 		for j in j_recipes_indexes[0]:
# 			recipe_i = reci_ingr_matrix[i,:]
# 			recipe_j = reci_ingr_matrix[j,:]
# 			#dot product calculates how mnay ingredients recipe_i and recipe_j have in common
# 			sim_sum += np.dot(recipe_i, recipe_j.T)
# 	r_similarity = float(sim_sum)/(len(i_recipes_indexes[0])*len(j_recipes_indexes[0]))
# 	#recipes contain i, recipes contain j, they share (r_similarity) ingredients on average
# 	return r_similarity

# def save_recipe_dictionary():
# 	"""
# 		store a dictionary in file where 
# 		the keys are indexes of recipes
# 		and the item is the list of indexes of ingredients that the recipe contains
# 	"""
# 	recipe_dict = defaultdict(list)
# 	for recipe_index in range(0,52375):
# 		items = np.where(reci_ingr_matrix[recipe_index,:] == 1)
# 		recipe_dict[recipe_index] = items[0]
# 	with open(recipe_dict_file, 'wb') as handle_:
# 		pickle.dump(recipe_dict, handle_)
# 	handle_.close()

# def save_ingredient_dictionary():
# 	"""
# 		store a dictionary as file where 
# 		the keys are indexes of ingredients
# 		and the item is the list of indexes of recipes that contain the ingredient
# 	"""
# 	ingredient_dict = defaultdict(list)
# 	for ingredient_index in range(0,248):
# 		items = np.where(reci_ingr_matrix[:,ingredient_index] == 1)
# 		ingredient_dict[ingredient_index] = items[0]
# 	with open(ingredient_dict_file, 'wb') as handle:
# 		pickle.dump(ingredient_dict, handle)
# 	handle.close()

def save_normalized_similarity_matrix(similarity_matrix_file):
	"""
	    compute the raw similarity between ingredients according to the recipes that contain them,
	    by the formular
	    .. math::
	       sim(i,j) = \frac{1}{n_I}\frac{1}{n_J}\sum_{I_{R_i} \in I}\sum_{ I_{R_j} \in J} <I_{R_i},I_{R_j}>.

	    Parameters
	    ----------
	    ingr_index_i : int
	        the index of the first ingredient
	    ingr_index_j : int
	        the index of the second ingredient

	    Returns
	    -------
	    r_similarity : double
	        The raw similarity score between two ingredients.
    """

	n_ingredients = len(ingr_all)
	similarity_matrix = np.zeros((248,248))
	result_queue = Queue.Queue()
	jobs =[]
	with open(recipe_dict_file, 'rb') as handle:
		recipe_dict = pickle.load(handle)
	# handle.close()
	with open(ingredient_dict_file, 'rb') as handle_:
		ingredient_dict = pickle.load(handle_)
	handle_.close()
	handle.close()
	try:
		for index_i in range(n_ingredients-1):
			t = Thread(target=normalized_similarity_row, args=('thread'+str(index_i), \
														index_i, \
														n_ingredients, \
														recipe_dict, \
														ingredient_dict))
			t.start()
			jobs.append(t)
	except:
		print "error: unable to start thread"

	for t in jobs:
		t.join()

	for _ in range(n_ingredients-1):
		result_dict = result_queue.get()		
		for key,value in result_dict.iteritems():
			rest = [0]*(key+1)
			matrix_row = rest + value
			similarity_matrix[int(key),:] = matrix_row

	similarity_matrix_symmetric = similarity_matrix + similarity_matrix.T #- np.diag(a.diagonal())
	np.savetxt(similarity_matrix_file, similarity_matrix_symmetric,fmt='%.9f')

def normalized_similarity_row(threadname,index_i,n_ingredients,recipe_dict,ingredient_dict):
	sim_row = list()
	for index_j in range(index_i+1,n_ingredients):
		sim = normalized_similarity(recipe_dict,ingredient_dict,index_i,index_j)
		sim_row.append(sim)
		print threadname+': index_i: '+str(index_i)+' index_j: '+str(index_j)+' similarity: '+str(sim)+'\n'
		# break
	result_queue.put({index_i:sim_row})
	# return sim_row

def normalized_similarity(recipe_dict,ingredient_dict,ingr_index_i,ingr_index_j):
	i_recipes_indexes = ingredient_dict[ingr_index_i]#[0]
	j_recipes_indexes = ingredient_dict[ingr_index_j]#[0]
	sim_sum = 0
	for i in i_recipes_indexes:
		for j in j_recipes_indexes:
			ingredient_set_i = recipe_dict[i]
			ingredient_set_j = recipe_dict[j]
			sim = 2*len(set(ingredient_set_i) & set(ingredient_set_j))/float(len(ingredient_set_i)+len(ingredient_set_j))
			sim_sum += sim
	r_similarity = float(sim_sum)/(len(i_recipes_indexes)*len(j_recipes_indexes))
	return r_similarity

# def save_raw_similarity_matrix(similarity_matrix_file):
# 	n_ingredients = len(ingr_all)
# 	similarity_matrix = np.zeros((248,248))
# 	for index_i in range(n_ingredients-1):
# 		for index_j in range(index_i+1,n_ingredients):
# 			similarity_matrix[index_i,index_j] = recipe_level_similarity(index_i,index_j)
# 			similarity_matrix[index_j,index_i] = similarity_matrix[index_i,index_j]
# 			print 'index_i:',index_i,'index_j:',index_j,'similarity:',similarity_matrix[index_i,index_j],similarity_matrix[index_j,index_i]
# 		np.savetxt(similarity_matrix_file, similarity_matrix,fmt='%.9f')

# def save_cosine_similarity_matrix():
# 	model = models.Word2Vec.load_word2vec_format(best_vector_file, binary=False)
# 	cosine_similarity_matrix = np.zeros((248,248))
# 	for i in range(0,248):
# 		for j in range(i+1,248):
# 			cosine_similarity_matrix[i,j] = model.similarity(ingr_all[i], ingr_all[j])
# 			cosine_similarity_matrix[j,i] = cosine_similarity_matrix[i,j]
# 		# print 'ingredient i:',ingr_all[i],'ingredient j:',ingr_all[j],'similarity:',cosine_similarity_matrix[i,j]
# 	np.savetxt(cosine_similarity_matrix_file, cosine_similarity_matrix,fmt='%.9f')
# 	plot_similarity_distribution(similarity_matrix_file=cosine_similarity_matrix_file,
# 											upper_lim=1,
# 											lower_lim=-1,
# 											interval=0.04)

# #only for the distribution, the index of ingredients are different from others
# def save_angular_similarity_matrix(): 
# 	vec_data,ingr_names = get_vec_from_model(best_vector_file, 10)
# 	angular_similarity_matrix = cosine_similarity(vec_data)
# 	angular_similarity_matrix_bounded = 1 - np.arccos(angular_similarity_matrix)/np.pi
# 	angular_similarity_matrix_bounded[np.isnan(angular_similarity_matrix_bounded)] = 1
	
# 	np.savetxt(angular_similarity_matrix_file, angular_similarity_matrix_bounded,fmt='%.9f')

# 	mask = np.ones(angular_similarity_matrix_bounded.shape, dtype=bool)
# 	np.fill_diagonal(mask, 0)
# 	max_value = angular_similarity_matrix_bounded[mask].max()
# 	min_value = angular_similarity_matrix_bounded.min()
# 	avg_value = np.average(angular_similarity_matrix_bounded)
# 	# print max_value,min_value,avg_value
# 	plot_similarity_distribution(similarity_matrix_file=angular_similarity_matrix_file,
# 											upper_lim=1,
# 											lower_lim=0,
# 											interval=0.02,
# 											max_value=max_value,
# 											min_value=min_value,
# 											avg_value=avg_value)

def avg_similarity_all():
	"""
		compute the average similarity among all 248 ingredients.

	"""
	n_ingredients = len(ingr_all)
	# print n_ingredients #248
	sim_tot = 0
	for index_i in range(n_ingredients):
		for index_j in range(index_i+1,n_ingredients):
			sim_tot += similarity_matrix[index_i,index_j]
		# print 'index_i',index_i,tot_sim
	avg_sim = 2*float(sim_tot)/(n_ingredients*(n_ingredients-1))
	return avg_sim

def agv_similarity_by_category():
	"""
		compute the average similarity inside each food categories.

	"""
	category_dict = defaultdict(list)
	similarities = list()
	for i in range(248):
		category = cate_all[i]
		category_dict[category].append(i)
	# print category_dict
	for key in category_dict.keys():
		sim_tot = 0
		ingr_list = category_dict[key]
		for i in range(len(ingr_list)):
			for j in range(i+1,len(ingr_list)):
				sim_tot += similarity_matrix[ingr_list[i], ingr_list[j]]
		similarity_in_category = 2*float(sim_tot)/(len(ingr_list)*(len(ingr_list)-1))
		similarities.append(similarity_in_category)
		print 'for',key,',the average ingredient similarity inside category is:',similarity_in_category
		plot_similarity_by_category(category_dict.keys(),similarities)

def similarity_in_cluster(cluster_items):
	"""
		Calculating similarity between ingredients
	"""
	sim_tot = 0
	item_num = len(cluster_items[0])
	# print item_num
	for item_i in range(item_num):
		for item_j in range(item_i+1,item_num):
			sim_tot += similarity_by_name(cluster_items[0][item_i], cluster_items[0][item_j])
	avg_sim = 2*float(sim_tot)/(item_num*(item_num-1))
	return avg_sim

def name2index(ingredient_name):
	index = list(ingr_all).index(ingredient_name)
	return index

def similarity_by_name(ingredient_name_i,ingredient_name_j):
	index_i = name2index(ingredient_name_i)
	index_j = name2index(ingredient_name_j)
	similarity = similarity_matrix[index_i,index_j]
	return similarity

def calculate_similarity(pred,ingredients,n_clusters):
	"""
		Given the cluster predictions,
		calculate the average similarity among ingredients in clusters
	"""
	# labels = np.row_stack((ingredients,pred))
	# pred_uniq = np.unique(labels[1])
	similarities = list()
	clusters = dict()
	for n in range(0,n_clusters): #0,1,2
		item_index = np.where(pred == n)
		# print item_index
		items = list()
		for index in item_index:
			items.append(ingredients[index])
		clusters[n] = items
		# print clusters[n]
		if len(items[0]) > 1:
			similarity = similarity_in_cluster(clusters[n])
			similarities.append(similarity)
		else:
			continue
		# break
	# print len(similarities)
	avg = reduce(lambda x, y: x + y, similarities) / len(similarities)
	return avg


def get_vec_from_model(filename,size):
	"""
		getting vector matrix from given file
	"""
	vec_data = np.genfromtxt(filename,
		delimiter=' ',
    	dtype=None,
    	usecols=(_ for _ in range (1,size+1)),
    	autostrip=True,skip_header=2)
	ingr_names = np.genfromtxt(filename,
		delimiter=' ',
		dtype=None, #248 major ingredients
		usecols=0,
		autostrip=True,skip_header=2)
	return vec_data,ingr_names
	
def sort_vector_index():
	pass

if __name__ == "__main__":
	main()
