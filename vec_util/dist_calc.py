import numpy as np
from scipy.spatial.distance import jaccard

ingr_reci_matrix = np.genfromtxt('/home/liah/Documents/stage/data/ingr_reci_matrix.txt',
	delimiter=' ',
	dtype=int,
	#usecols=(0),
	autostrip=True,skip_header=0)

cooccur_matrix = np.genfromtxt('/home/liah/Documents/stage/data/co_occur_matrix.txt',
	delimiter=' ',
	dtype=int,
	#usecols=(0),
	autostrip=True,skip_header=0)

def main():
	pass
	#matrix_gen()	
	#print calculate_jaccard(24, 262)

def matrix_gen():
	ingr_all = np.genfromtxt('/home/liah/Documents/stage/data/ingredients.csv',
		delimiter=',',
		dtype=None,
		usecols=(0),
		autostrip=True,skip_header=0)
	ingr_reci_matrix = np.zeros((381,55001),dtype=int)
	i=0
	with open('/home/liah/Documents/stage/data/recipes_as_input.txt','r') as f:
	 	for line in f.readlines():
	 		ingredients = line.strip('\n').split(' ');
			for ingredient in ingredients:
				index = list(ingr_all).index(ingredient)
				ingr_reci_matrix[index,i] = 1
			i += 1
		#break
	f.close()
	np.savetxt('/home/liah/Documents/stage/data/ingr_reci_matrix.txt', ingr_reci_matrix, fmt='%d')

def cooccurance_gen():
	int_matrix = ingr_reci_matrix.astype(int)
	co_occur_matrix = int_matrix.dot(int_matrix.T)
	#print co_occur_matrix.shape
	#print co_occur_matrix[12,:]
	np.savetxt('./co_occur_matrix.txt', co_occur_matrix,fmt='%d')

'''given ingredient indexes index_u and index_v, 
	calculate the cooccurance times of these two ingredients. '''

def calculate_cooccurance(index_u,index_v):
	co_occur = cooccur_matrix[index_u,index_v]
	return co_occur


'''given ingredient indexes index_u and index_v, 
	calculate the jaccard distance between these two ingredients. '''

def calculate_jaccard(index_u, index_v):
	u = ingr_reci_matrix[index_u,:]
	v = ingr_reci_matrix[index_v,:]
	# print len(np.unique(u))
	# print len(np.unique(v))
	dist = jaccard(u,v)
	return dist

if __name__ == "__main__":
	main()