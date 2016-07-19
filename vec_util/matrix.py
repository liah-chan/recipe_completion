import numpy as np
from scipy.spatial.distance import jaccard
import os

# ingr_reci_matrix = np.genfromtxt('/home/liah/Documents/stage/data/ingr_reci_matrix.txt',
# 	delimiter=' ',
# 	dtype=int,
# 	#usecols=(0),
# 	autostrip=True,skip_header=0)

# cooccur_matrix = np.genfromtxt('/home/liah/Documents/stage/data/co_occur_matrix.txt',
# 	delimiter=' ',
# 	dtype=int,
# 	#usecols=(0),
# 	autostrip=True,skip_header=0)
ingr_reci_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingr_reci_matrix.txt'))
reci_ingr_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/reci_ingr_matrix.txt'))
co_occur_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/co_occur_matrix.txt'))
common_ingr_matrix_file = os.path.realpath(os.path.join(os.getcwd(),'../data/common_ingr_matrix.txt'))

def main():
	ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))
	recipe_file = os.path.realpath(os.path.join(os.getcwd(),'../data/recipes_major.txt'))
	cooccurance_gen(ingredient_file,recipe_file)
	#print calculate_jaccard(24, 262)

'''
save ingr_reci_matrix and reci_ingr_matrix if matrix file not exist
'''
def matrix_gen(ingredient_file,recipe_file):
	ingr_all = np.genfromtxt(ingredient_file,
					delimiter=',',
					dtype=None,
					usecols=(0),
					autostrip=True,skip_header=0)

	n_ingredient = len(ingr_all)
	# print n_ingredient
	with open(recipe_file) as f:
		n_recipe = sum(1 for _ in f)
	f.close()
	# print n_ingredient,n_recipe

	ingr_reci_matrix = np.zeros((n_ingredient,n_recipe),dtype=int)
	reci_ingr_matrix = np.zeros((n_recipe,n_ingredient),dtype=int)

	i=0
	with open(recipe_file,'r') as f:
	 	for line in f.readlines():
	 		ingredients = line.strip('\n').split(' ');
			for ingredient in ingredients:
				index = list(ingr_all).index(ingredient)
				ingr_reci_matrix[index,i] = 1
				# reci_ingr_matrix[i,index] = 1
			i += 1
		#break
	f.close()
	np.savetxt(ingr_reci_matrix_file, ingr_reci_matrix, fmt='%d')
	np.savetxt(reci_ingr_matrix_file, ingr_reci_matrix.T, fmt='%d')

def get_common_ingr_matrix(ingredient_file,recipe_file):
	if not os.path.isfile(common_ingr_matrix_file):
		pass
		# common_ingr_gen(ingredient_file, recipe_file)
	common_ingr_matrix = np.genfromtxt(common_ingr_matrix_file,
							delimiter=' ',
							dtype=int,
							#usecols=(0),
							autostrip=True,skip_header=0)
	return common_ingr_matrix

def common_ingr_gen(ingredient_file,recipe_file):
	if not os.path.isfile(reci_ingr_matrix_file):
		matrix_gen(ingredient_file, recipe_file)

	reci_ingr_matrix = np.genfromtxt(reci_ingr_matrix_file,
							delimiter=' ',
							dtype=int,
							#usecols=(0),
							autostrip=True,skip_header=0)
	int_matrix = reci_ingr_matrix.astype(int)
	common_ingr_matrix = int_matrix.dot(int_matrix.T)
	np.savetxt(common_ingr_matrix_file, common_ingr_matrix,fmt='%d')

def get_co_occur_matrix(ingredient_file,recipe_file):
	if not os.path.isfile(co_occur_matrix_file):
		cooccurance_gen(ingredient_file,recipe_file)

	co_occur_matrix = np.genfromtxt(co_occur_matrix_file,
							delimiter=' ',
							dtype=int,
							#usecols=(0),
							autostrip=True,skip_header=0)
	return co_occur_matrix

def cooccurance_gen(ingredient_file,recipe_file):
	if not os.path.isfile(ingr_reci_matrix_file):
		matrix_gen(ingredient_file, recipe_file)

	ingr_reci_matrix = np.genfromtxt(ingr_reci_matrix_file,
							delimiter=' ',
							dtype=int,
							#usecols=(0),
							autostrip=True,skip_header=0)
	int_matrix = ingr_reci_matrix.astype(int)
	co_occur_matrix = int_matrix.dot(int_matrix.T)
	np.savetxt(co_occur_matrix_file, co_occur_matrix,fmt='%d')

# '''given ingredient indexes index_u and index_v, 
# 	calculate the cooccurance times of these two ingredients. '''

# def calculate_cooccurance(index_u,index_v):
# 	co_occur = cooccur_matrix[index_u,index_v]
# 	return co_occur


# '''given ingredient indexes index_u and index_v, 
# 	calculate the jaccard distance between these two ingredients. '''

# def calculate_jaccard(index_u, index_v):
# 	u = ingr_reci_matrix[index_u,:]
# 	v = ingr_reci_matrix[index_v,:]
# 	# print len(np.unique(u))
# 	# print len(np.unique(v))
# 	dist = jaccard(u,v)
# 	return dist

if __name__ == "__main__":
	main()