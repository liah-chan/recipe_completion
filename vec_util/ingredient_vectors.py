#system module
import re
import numpy as np
import os
import sys
from collections import defaultdict
from gensim.models import word2vec as w2v
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.spatial.distance import dice
from scipy.spatial.distance import pdist,squareform
#local module
sys.path.append('../vec_util/')
from matrix import get_common_ingr_matrix,get_co_occur_matrix
from plotting import plot_similarity_by_category,plot_similarity_distribution,plot_similarity
import pickle

ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))
recipe_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipes_major.txt'))
reci_ingr_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/reci_ingr_matrix.txt'))

raw_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/raw_similarity_matrix.txt'))
normalized_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/normalized_similarity_matrix.txt'))
# cosine_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/cosine_similarity_matrix.txt'))
euclidean_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/euclidean_similarity_matrix'))
angular_similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/angular_similarity_matrix'))

# recipe_similarity_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipe_similarity_list.pickle'))

ingredient_dict_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredient_dict.pickle'))
recipe_dict_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipe_dict.pickle'))

# best_vector_file = os.path.realpath(os.path.join(os.getcwd(),'../tune_size/model_10'))
best_vector_file = os.path.realpath(os.path.join(os.getcwd(),'../tune_size/best_vec_sorted.txt'))

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
# similarity_matrix = np.genfromtxt(similarity_matrix_file,   ######
# 		delimiter=' ',
# 		dtype=float,
# 		#usecols=(0),
# 		autostrip=True,skip_header=0)

def main():
	find_best_vec()
	# vector_file = os.path.realpath(os.path.join(os.getcwd(),'../tune_size/new/model_10'))
	# save_euclidean_similarity_matrix(vector_file, size=10, skip_header=0)

	# plot_normalized_similarity(normalized_similarity_matrix_file)
	# agv_similarity_by_category(raw_similarity_matrix_file)


def get_reci_ingr_matrix():
	if not os.path.isfile(reci_ingr_matrix_file):  ######
		matrix_gen(ingredient_file, recipe_file)
	#read recipe ingredient matrix (52375, 248) from file
	reci_ingr_matrix = np.genfromtxt(reci_ingr_matrix_file,   ######
		delimiter=' ',
		dtype=int,
		#usecols=(0),
		autostrip=True,skip_header=0)
	return reci_ingr_matrix
	# print reci_ingr_matrix.shape #(52375, 248)

def get_normalized_similarity_matrix():
	if not os.path.isfile(normalized_similarity_matrix_file):
		save_normalized_similarity_matrix(normalized_similarity_matrix_file)
	normalized_similarity_matrix = np.genfromtxt(normalized_similarity_matrix_file,
										delimiter=' ',
										dtype=float,
										#usecols=(0),
										autostrip=True,skip_header=0)
	return normalized_similarity_matrix

def raw_similarity(ingr_index_i,ingr_index_j):
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

	i_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_i] == 1)
	j_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_j] == 1)
	# print i_recipes_indexes,j_recipes_indexes
	sim_sum = 0
	for i in i_recipes_indexes[0]:
		for j in j_recipes_indexes[0]:
			recipe_i = reci_ingr_matrix[i,:]
			recipe_j = reci_ingr_matrix[j,:]
			#dot product calculates how mnay ingredients recipe_i and recipe_j have in common
			sim_sum += np.dot(recipe_i, recipe_j.T)
	r_similarity = float(sim_sum)/(len(i_recipes_indexes[0])*len(j_recipes_indexes[0]))
	#recipes contain i, recipes contain j, they share (r_similarity) ingredients on average
	return r_similarity

def save_raw_similarity_matrix(similarity_matrix_file):
	"""
		Calculate raw similarity between all ingredient pairs and save to a file
	"""
	n_ingredients = len(ingr_all)
	similarity_matrix = np.zeros((248,248))
	for index_i in range(n_ingredients-1):
		for index_j in range(index_i+1,n_ingredients):
			similarity_matrix[index_i,index_j] = raw_similarity(index_i,index_j)
			similarity_matrix[index_j,index_i] = similarity_matrix[index_i,index_j]
			print 'index_i:',index_i,'index_j:',index_j,'similarity:',similarity_matrix[index_i,index_j],similarity_matrix[index_j,index_i]
		np.savetxt(similarity_matrix_file, similarity_matrix,fmt='%.9f')

def save_recipe_dictionary():
	"""
		store a dictionary in file where 
		the keys are indexes of recipes
		and the item is the list of indexes of ingredients that the recipe contains
	"""
	reci_ingr_matrix = get_reci_ingr_matrix()
	recipe_dict = defaultdict(list)
	for recipe_index in range(0,52375):
		items = np.where(reci_ingr_matrix[recipe_index,:] == 1)
		recipe_dict[recipe_index] = items[0]
	with open(recipe_dict_file, 'wb') as handle_:
		pickle.dump(recipe_dict, handle_)
	handle_.close()

def save_ingredient_dictionary():
	"""
		store a dictionary as file where 
		the keys are indexes of ingredients
		and the item is the list of indexes of recipes that contain the ingredient
	"""
	reci_ingr_matrix = get_reci_ingr_matrix()
	ingredient_dict = defaultdict(list)
	for ingredient_index in range(0,248):
		items = np.where(reci_ingr_matrix[:,ingredient_index] == 1)
		ingredient_dict[ingredient_index] = items[0]
	with open(ingredient_dict_file, 'wb') as handle:
		pickle.dump(ingredient_dict, handle)
	handle.close()

def plot_normalized_similarity(similarity_matrix_file):
	if not os.path.isfile(similarity_matrix_file):
		save_normalized_similarity_matrix(similarity_matrix_file)
	normalized_similarity_matrix = np.genfromtxt(similarity_matrix_file,
										delimiter=' ',
										dtype=float,
										#usecols=(0),
										autostrip=True,skip_header=0)
	mask = np.ones(normalized_similarity_matrix.shape, dtype=bool)
	np.fill_diagonal(mask, 0)
	max_value = normalized_similarity_matrix[mask].max()
	min_value = normalized_similarity_matrix[mask].min()
	avg_value = np.average(normalized_similarity_matrix[mask])
	plot_similarity_distribution(similarity_matrix_file=similarity_matrix_file, 
										metric='extended_normalized', 
										upper_lim=0.5, 
										lower_lim=0.0, 
										interval=0.01, 
										max_value=max_value, 
										min_value=min_value, 
										avg_value=avg_value)
	
	

def save_normalized_similarity_matrix(similarity_matrix_file):
	"""
		calculate and save normalized similarity scores of ingredient pairs
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
	"""
		given an ingredient index number, calculate the similarity score between this ingredient
		and ingredients with index larger than it, 
		and store the result into queue

		Parameters
	    ----------
	    threadname : str
	    	name of the thread involking this function
	    index_i : int
	        the index of the first ingredient
	    n_ingredients : int
	    	total number of ingredients
	    recipe_dict : dict
	    	dictionary with recipe indexes as keys,
	    	and lits of ingredients in that recipe as item
	    ingredient_dict : dict
	    	dictionary with ingredient indexes as keys,
	    	and lits of recipes that contain the ingredient as item
	"""
	sim_row = list()
	for index_j in range(index_i+1,n_ingredients):
		sim = normalized_similarity(recipe_dict,ingredient_dict,index_i,index_j)
		sim_row.append(sim)
		print threadname+': index_i: '+str(index_i)+' index_j: '+str(index_j)+' similarity: '+str(sim)+'\n'
		# break
	result_queue.put({index_i:sim_row})
	# return sim_row

def normalized_similarity(recipe_dict,ingredient_dict,ingr_index_i,ingr_index_j):
	"""
	    compute the normalized similarity between ingredients according to the recipes that contain them,
	    by the formular
	    .. math::
	       sim(i,j) = \frac{2 \cdot \sum_{I_{R_i} \in I} \sum_{ I_{R_j} \in J} |I_{R_i} \cap I_{R_j}|}{\sum_{I_{R_i} \in I} |I_{R_i}|
	        + \sum_{I_{R_j} \in J} |I_{R_j}|}.

	    Parameters
	    ----------
	    recipe_dict : dict
	    	dictionary with recipe indexes as keys,
	    	and lits of ingredients in that recipe as item
	    ingredient_dict : dict
	    	dictionary with ingredient indexes as keys,
	    	and lits of recipes that contain the ingredient as item
	    ingr_index_i : int
	        the index of the first ingredient
	    ingr_index_j : int
	        the index of the second ingredient

	    Returns
	    -------
	    r_similarity : double
	        The normalized similarity score between two ingredients.
    """
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

# def save_recipe_similarity():
# 	with open(recipe_dict_file, 'rb') as handle:
# 		recipe_dict = pickle.load(handle)
# 	handle.close()
# 	# for key in recipe_dict.keys():
# 	# matrix = np.zeros((52375,52375),dtype=int)
# 	similarity_list = list()
# 	length = len(recipe_dict.keys()) #52375
# 	# for i in range(length-1):
# 	for i in range(0,1):
# 		for j in range(i+1,length):
# 			# print  set(recipe_dict[i][0]) & set(recipe_dict[j][0])
# 			sim = round(len(set(recipe_dict[i]) & set(recipe_dict[j]))/float(len(recipe_dict[i])+len(recipe_dict[j])),3)
# 			similarity_list.append(sim)
# 			# matrix[j,i] = matrix[i,j]
# 			# print matrix[i,j]
# 			# print sim
# 			# break
# 		# break
# 	with open(recipe_similarity_file, 'wb') as handle_:
# 		pickle.dump(similarity_list, handle_)
# 	handle_.close()
# 	# matrix_symmetric = matrix + matrix.T
# 	# np.savetxt(recipe_similarity_matrix_file, matrix, fmt='%d')	

# def save_cosine_similarity_matrix():
# 	model = models.Word2Vec.load_word2vec_format(best_vector_file, binary=False)
# 	cosine_similarity_matrix = np.zeros((248,248))
# 	for i in range(0,248):
# 		for j in range(i+1,248):
# 			cosine_similarity_matrix[i,j] = model.similarity(ingr_all[i], ingr_all[j])
# 			cosine_similarity_matrix[j,i] = cosine_similarity_matrix[i,j]
# 		# print 'ingredient i:',ingr_all[i],'ingredient j:',ingr_all[j],'similarity:',cosine_similarity_matrix[i,j]
# 	np.savetxt(cosine_similarity_matrix_file, cosine_similarity_matrix,fmt='%.9f')

# 	mask = np.ones(cosine_similarity_matrix.shape, dtype=bool)
# 	np.fill_diagonal(mask, 0)
# 	max_value = cosine_similarity_matrix[mask].max()
# 	min_value = cosine_similarity_matrix[mask].min()
# 	avg_value = np.average(cosine_similarity_matrix[mask])

# 	plot_similarity_distribution(similarity_matrix_file=cosine_similarity_matrix_file,
# 											metric='cosine',
# 											upper_lim=1,
# 											lower_lim=-1,
# 											interval=0.04,
# 											max_value=max_value,
# 											min_value=min_value,
# 											avg_value=avg_value)

def find_best_vec():
	"""
	find the vector file that analogs most with the normalized similarity
	"""
	similarity_list = list()
	for size in range(5,101,5):
		vector_file = os.path.realpath(os.path.join(os.getcwd(),'../tune_size/new/model_'+str(size)))
		angular_similarity_file = save_angular_similarity_matrix(vector_file, size, skip_header=0,metric_str='angular'+str(size))
		# print 'vec_size:',size,':'
		sim = compare_similarity(normalized_similarity_matrix_file, angular_similarity_file)
		similarity_list.append(sim)
	# plot_similarity(list(range(5,101,5)), similarity_list, 'size',upper_lim=150,lower_lim=0)


def save_angular_similarity_matrix(vector_file,size,skip_header,metric_str='angular'): 

	sorted_vec_file = sort_vector(vector_file)

	vec_data,ingr_names = get_vec_from_model(sorted_vec_file, size, skip_header)
	angular_similarity_matrix = cosine_similarity(vec_data)
	angular_similarity_matrix_bounded = 1 - np.arccos(angular_similarity_matrix)/np.pi
	angular_similarity_matrix_bounded[np.isnan(angular_similarity_matrix_bounded)] = 1

	similarity_file = angular_similarity_matrix_file+'_'+metric_str
	np.savetxt(similarity_file, angular_similarity_matrix_bounded,fmt='%.9f')

	# plot_angular_similarity(angular_similarity_matrix_bounded,metric_str)

	return similarity_file

def save_euclidean_similarity_matrix(vector_file,size,skip_header,metric_str='euclidean'): 

	sorted_vec_file = sort_vector(vector_file)

	vec_data,ingr_names = get_vec_from_model(sorted_vec_file, size, skip_header)
	euclidean_similarity_matrix = euclidean_distances(vec_data)
	# euclidean_similarity_matrix_bounded = 1 - np.arccos(angular_similarity_matrix)/np.pi
	# euclidean_similarity_matrix_bounded[np.isnan(angular_similarity_matrix_bounded)] = 1

	similarity_file = euclidean_similarity_matrix_file+'_'+metric_str
	np.savetxt(similarity_file, euclidean_similarity_matrix,fmt='%.9f')

	# plot_angular_similarity(angular_similarity_matrix_bounded,metric_str)

	return similarity_file

def plot_angular_similarity(matrix,metric_str):
	masked_matrix = masking_diagnol(matrix)
	max_value = masked_matrix.max()
	min_value = masked_matrix.min()
	avg_value = np.average(masked_matrix)
	# print max_value,min_value,avg_value
	plot_similarity_distribution(similarity_matrix_file=angular_similarity_matrix_file,
											metric=metric_str,
											upper_lim=1,
											lower_lim=0,
											interval=0.02,
											max_value=max_value,
											min_value=min_value,
											avg_value=avg_value)


def compare_similarity(matrix_file_1, matrix_file_2):
	"""
	given two similarity matix of the ingredients,
	compare if they agree with each other.
	i.e. 
	if according to 1st matrix, the most close ingredient for beef is pork,
	and according to 2nd matrix, the same, 
	then they agree with each other.
	"""
	ingr_matrix = np.empty((248,248),dtype=object)
	for i in range(0,248):
		ingr_matrix[i,:] = ingr_all
	# print ingr_matrix.shape
	# print ingr_matrix[2,2]

	matrix_1 = np.genfromtxt(matrix_file_1,
		delimiter=' ',
		dtype=float,
		autostrip=True,
		skip_header=0)

	matrix_2 = np.genfromtxt(matrix_file_2,
	delimiter=' ',
	dtype=float,
	autostrip=True,
	skip_header=0)
	# print matrix_1.shape,matrix_2.shape

	sorted_1 = np.argsort(-matrix_1,axis=1)
	sorted_2 = np.argsort(-matrix_2,axis=1)

	summ = 0
	for i in range(0,248):
		# if sorted_1[i,1] in sorted_2[i,1:16]:
		# 	summ+=1
		common = len(set(sorted_1[i,1:16]) & set(sorted_2[i,1:16]))
		summ+=common/16.0
	index = list(ingr_all).index('beef')
	# print ingr_all[index]
	# print ingr_all[sorted_1[index,1:16]]
	# print ingr_all[sorted_2[index,1:16]]
	# print summ/248.0
	return summ/248.0


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

def masking_diagnol(matrix):
	mask = np.ones(matrix.shape,dtype=bool)
	np.fill_diagonal(mask, 0)
	return matrix[mask]



def agv_similarity_by_category(similarity_matrix_file):
	"""
		compute the average similarity inside each food categories.

	"""
	similarity_matrix = np.genfromtxt(similarity_matrix_file,   ######
		delimiter=' ',
		dtype=float,
		#usecols=(0),
		autostrip=True,skip_header=0)
	masked_matrix = masking_diagnol(similarity_matrix)
	avg_value = np.average( masked_matrix )
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
		plot_similarity_by_category(category_dict.keys(),similarities,'',avg_value)

def similarity_in_cluster(cluster_items,normalized_similarity_matrix):
	"""
		Calculating similarity between ingredients
	"""
	sim_tot = 0
	item_num = len(cluster_items[0])
	# print item_num
	for item_i in range(item_num):
		for item_j in range(item_i+1,item_num):
			# sim_tot += similarity_by_name(cluster_items[0][item_i], cluster_items[0][item_j],normalized_similarity_matrix)
			index_i = name2index(cluster_items[0][item_i])
			index_j = name2index(cluster_items[0][item_j])
			similarity = normalized_similarity_matrix[index_i,index_j]
			sim_tot += similarity
	avg_sim = 2*float(sim_tot)/(item_num*(item_num-1))
	return avg_sim

def name2index(ingredient_name):
	index = list(ingr_all).index(ingredient_name)
	return index

# def similarity_by_name(ingredient_name_i,ingredient_name_j,normalized_similarity_matrix):

# 	index_i = name2index(ingredient_name_i)
# 	index_j = name2index(ingredient_name_j)
# 	similarity = normalized_similarity_matrix[index_i,index_j]
# 	return similarity

def calculate_similarity(pred,ingredients,n_clusters):
	"""
		Given the cluster predictions,
		calculate the average similarity among ingredients in clusters
	"""
	# labels = np.row_stack((ingredients,pred))
	# pred_uniq = np.unique(labels[1])
	normalized_similarity_matrix = get_normalized_similarity_matrix()

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
			similarity = similarity_in_cluster(clusters[n],normalized_similarity_matrix)
			similarities.append(similarity)
		else:
			continue
		# break
	# print len(similarities)
	avg = reduce(lambda x, y: x + y, similarities) / len(similarities)
	return avg


def get_vec_from_model(filename,size,skip_header):
	"""
		getting vector matrix from given file
	"""
	vec_data = np.genfromtxt(filename,
		delimiter=' ',
    	dtype=None,
    	usecols=(_ for _ in range (1,size+1)),
    	autostrip=True,skip_header=skip_header)
	ingr_names = np.genfromtxt(filename,
		delimiter=' ',
		dtype=None, #248 major ingredients
		usecols=0,
		autostrip=True,skip_header=skip_header)
	return vec_data,ingr_names
	
def sort_vector(model_file):

	# ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))
	# model_file = './model_10'
	directory = os.path.dirname(model_file)
	filename = os.path.basename(model_file)
	sorted_vec_file = os.path.realpath(os.path.join(directory,filename+'sorted.txt'))

	with open(ingredient_file,'r+') as f1, open(model_file,'r+') as f2, open(sorted_vec_file,'w+') as tf:
		vector_list = f2.readlines()
		# ingredient_list = f1.readlines()
		for line in f1.readlines():
			ingredient = line.split(',')[0]
			# vec = str([item.startswith(ingredient) for item in vector_list])
			for item in vector_list:
				ingr = item.split(' ')[0]
				# print ingr
				if ingr == ingredient:
					vec = item
					vector_list.remove(item)
				
			# for item in vector_list
			tf.write(vec)
	tf.close()
	f2.close()
	f1.close()

	return sorted_vec_file

def save_network_encoding(ingredient_file,recipe_file):
	co_occur_matrix = get_co_occur_matrix(ingredient_file, recipe_file)
	network_encoing_file = os.path.realpath(os.path.join(os.getcwd(),'../data/network_encoding.txt'))
	with open(network_encoing_file,'a+') as f:
		for i in range(len(ingr_all)): #248
			for j in range(len(ingr_all)):
				if i == j:
					continue
				string = ingr_all[i]+'\t'+ingr_all[j]+'\t'+str(co_occur_matrix[i,j])+'\n'+ \
							ingr_all[j]+'\t'+ingr_all[i]+'\t'+str(co_occur_matrix[i,j])+'\n'
				f.write(string)
			# 	break
			# break
	f.close()

if __name__ == "__main__":
	main()
