# import multiprocessing
import numpy as np
from sklearn import cluster
import sys
import os
sys.path.append('../vec_util/')
from ingredient_vectors import calculate_similarity,get_vec_from_model
from plotting import plot_similarity

# size = 10
n_clusters = 9
ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))

def main():
	parameters = list(range(5,101,5))
	similarities = list()
	for size in range(5,101,5):
		y = compare_by_cluster(size)
		# y = compare_by_class(size)
		similarities.append(y)
	# print similarities
	plot_similarity(parameters, similarities, para_name='vector_size')

def compare_by_class(size):
	class_file = os.path.realpath(os.path.join(os.getcwd(),'./classes_model_'+str(size))) ######
	ingredients = np.genfromtxt(class_file,
		delimiter=' ',
		dtype=None,
		usecols=0,
		skip_header=1,
		autostrip=True)
	pred = np.genfromtxt(class_file,
		delimiter=' ',
		dtype=None,
		usecols=1,
		skip_header=1,
		autostrip=True)
	y = calculate_similarity(pred,ingredients,n_clusters)	
	# break
	return y

def compare_by_cluster(size):
	model_file = os.path.realpath(os.path.join(os.getcwd(),'./model_'+str(size))) ######
	X,ingredients = get_vec_from_model(filename=model_file,size=size)			
	model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
		linkage='average',affinity="cosine")
	#pred stores the cluster label of each ingredient, starting from 0
	pred = model.fit(X).labels_ 
	# print np.unique(pred)
	y = calculate_similarity(pred,ingredients,n_clusters)
	
	# break
	return y

if __name__ == "__main__":
    main()