import numpy as np
import os
import sys
from collections import defaultdict
sys.path.append('../vec_util/')
from matrix import get_common_ingr_matrix
from plotting import plot_similarity_by_category


ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))
recipe_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipes_major.txt'))
reci_ingr_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/reci_ingr_matrix.txt'))
vec_file = ''
similarity_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/similarity_matrix.txt'))

if not os.path.isfile(reci_ingr_matrix_file):
	matrix_gen(ingredient_file, recipe_file)

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
	# avg_sim = cal_original_similarity()
	# avg_sim_1 = recipe_level_similarity(1, 2)
	# avg_sim_2 = recipe_level_similarity(20, 50)
	# print avg_sim_1,avg_sim_2
	# similarity_matrix_gen(similarity_matrix_file)
	agv_similarity_by_category()
	# print 'the average similarity among all ingredients:',avg_similarity_all()

"""
compute similarity between ingredients according to the recipes that contains them.

"""
def recipe_level_similarity(ingr_index_i,ingr_index_j):
	i_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_i] == 1)
	j_recipes_indexes = np.where(reci_ingr_matrix[:,ingr_index_j] == 1)
	# print i_recipes_indexes,j_recipes_indexes
	sim_sum = 0
	for i in i_recipes_indexes[0]:
		for j in j_recipes_indexes[0]:
			# sim_sum += common_ingr_matrix[i][j]
			# print i,j
			recipe_i = reci_ingr_matrix[i,:]
			recipe_j = reci_ingr_matrix[j,:]
			#dot product calculates how mnay ingredients recipe_i and recipe_j have in common
			sim_sum += np.dot(recipe_i, recipe_j.T)
			# print sim_sum
			# break
		# break
	r_similarity = float(sim_sum)/(len(i_recipes_indexes[0])*len(j_recipes_indexes[0]))
	#recipes contain i, recipes contain j, they share r_similarity ingredients on average
	return r_similarity

def similarity_matrix_gen(similarity_matrix_file):
	n_ingredients = len(ingr_all)
	similarity_matrix = np.zeros((248,248))
	for index_i in range(n_ingredients-1):
		for index_j in range(index_i+1,n_ingredients):
			similarity_matrix[index_i,index_j] = recipe_level_similarity(index_i,index_j)
			similarity_matrix[index_j,index_i] = similarity_matrix[index_i,index_j]
			print 'index_i:',index_i,'index_j:',index_j,'similarity:',similarity_matrix[index_i,index_j],similarity_matrix[index_j,index_i]
			# break
		# break
	np.savetxt(similarity_matrix_file, similarity_matrix,fmt='%.9f')
	# return avg_sim

"""
compute the average similarity among all 248 ingredients.

"""
def avg_similarity_all():
	n_ingredients = len(ingr_all)
	# print n_ingredients #248
	sim_tot = 0
	for index_i in range(n_ingredients):
		for index_j in range(index_i+1,n_ingredients):
			sim_tot += similarity_matrix[index_i,index_j]
		# print 'index_i',index_i,tot_sim
	avg_sim = 2*float(sim_tot)/(n_ingredients*(n_ingredients-1))
	return avg_sim

"""
compute the average similarity inside each food categories.

"""
def agv_similarity_by_category():
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



"""
routines for calculating similarity between ingredients
"""
def name2index(ingredient_name):
	index = list(ingr_all).index(ingredient_name)
	return index

def similarity_by_name(ingredient_name_i,ingredient_name_j):
	index_i = name2index(ingredient_name_i)
	index_j = name2index(ingredient_name_j)
	similarity = similarity_matrix[index_i,index_j]
	return similarity

def similarity_in_cluster(cluster_items):
	sim_tot = 0
	item_num = len(cluster_items[0])
	# print item_num
	for item_i in range(item_num):
		for item_j in range(item_i+1,item_num):
			sim_tot += similarity_by_name(cluster_items[0][item_i], cluster_items[0][item_j])
	avg_sim = 2*float(sim_tot)/(item_num*(item_num-1))
	return avg_sim

def calculate_similarity(pred,ingredients,n_clusters):
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
	#size=parameter
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
	
if __name__ == "__main__":
	main()
