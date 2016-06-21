# import multiprocessing
import numpy as np
from sklearn import cluster
import sys
import os
sys.path.append('../vec_util/')
from ingredient_vectors import calculate_similarity,get_vec_from_model
from plotting import plot_similarity

size = 10
n_clusters = 9
ingredient_file = os.path.realpath(os.path.join(os.getcwd(),'../data/ingredients_major.txt'))

#store 248 major ingredients and their food category into arrays 
ingr_all,cate_all = np.genfromtxt(ingredient_file, #248 major ingredients
		delimiter=',',
		dtype=None,
		usecols=(0,1),
		autostrip=True,
		skip_header=0,
		unpack=True)

def main():
	parameters = list(range(1,5))
	similarities = list()
	for model in range(1,5):	
		model_file = os.path.realpath(os.path.join(os.getcwd(),'./model_'+str(model))) ######
		X,ingredients = get_vec_from_model(filename=model_file,size=size)			
		model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
			linkage='average',affinity="cosine")
		#pred stores the cluster label of each ingredient, starting from 0
		pred = model.fit(X).labels_ 
		# print np.unique(pred)
		y = calculate_similarity(pred,ingredients,n_clusters)
		similarities.append(y)
		# break

	# print similarities
	plot_similarity(parameters, similarities, para_name='model')


if __name__ == "__main__":
    main()