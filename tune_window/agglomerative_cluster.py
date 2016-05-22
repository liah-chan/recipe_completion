import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict
import imp
import sys
sys.path.append('../vec_util/')
from dist_calc import calculate_jaccard,calculate_cooccurance
from sklearn.metrics import silhouette_score,silhouette_samples

size = 5
vol_num = 0

def main():
	best_scores = np.zeros((30,2))
	for window in range(3,33):
		model_file = './model_'+str(window)
		#model_file = './model_'
		X,ingredients = get_vec_from_model(filename=model_file,size=size)
		labels=list(ingredients)
		#Z = linkage(X,method='average',metric='cosine')
		# save_dendrogram(parameter=window, Z=Z, labels=labels)


		scores = np.zeros((12,2))#,dtype=[('n_cluster','i4'),('score','f8')])
		for n_clusters in range(3,15):
			cluster_indexes = fcluster(Z,n_clusters,criterion='maxclust')
			#print cluster_indexes
			score = silhouette_score(X, cluster_indexes,metric='cosine')
			scores[n_clusters-3,0]=n_clusters
			scores[n_clusters-3,1]=score
			#break
		best = scores[np.argmax(scores[:,1]),:]
		best_scores[window-3,:]=best
		print best_scores[window-3,:]
		#break
	best_window_size = np.argmax(best_scores[:,1])+3
	best_n_cluster = int(best_scores[np.argmax(best_scores[:,1]),0])
	print 'window size:',best_window_size,'n_cluster:',best_n_cluster

	model_file = './model_'+str(best_window_size)
	X,ingredients = get_vec_from_model(filename=model_file,size=size)
	Z = linkage(X,method='average',metric='cosine')
	save_dendrogram(parameter=best_window_size, Z=Z, labels=labels)
	cluster_indexes = fcluster(Z,best_n_cluster,criterion='maxclust')
	#print cluster_indexes
	# avg_dist,avg_cooccurs = calc_dist(cluster_indexes=cluster_indexes,
	# 	ingredients=ingredients)
	# print avg_dist
	# print avg_cooccurs


def save_dendrogram(parameter,Z,labels):
	plt.figure(figsize=(20, 50))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.ylabel('sample index')
	plt.xlabel('distance')
	dendrogram(
		Z,
		#truncate_mode='lastp',
		#p=3,
		labels=labels,
		orientation='left',
		#leaf_rotation=90.,  # rotates the x axis labels
		leaf_font_size=8.,  # font size for the x axis labels
	)
	for i in np.arange(0.4,1.0,0.1):
		plt.axvline(x=i, c='k',linestyle='dashed')
	#plt.show()
	plt.savefig('./dendrogram_'+str(parameter)+'_labeled.png')

def calc_dist(cluster_indexes,ingredients):
	d = defaultdict(list)
	for k in range(len(cluster_indexes)):
		d[cluster_indexes[k]].append(ingredients[k])
	for key,listitem in d.iteritems():
		print "%s: %s" %(key,listitem)
	inner_dist = list()
	inner_cooccur = list()
	dists = list()
	cooccurs = list()
	for key in d.keys():
		cluster_items = d.get(key)
		for i in range(len(cluster_items)):
			for j in range(len(cluster_items)):
				if i!=j:
					index_u = list(ingredients).index(cluster_items[i])
					index_v = list(ingredients).index(cluster_items[j])
					jac_dist = 1 - calculate_jaccard(index_u, index_v)
					cooccurance = calculate_cooccurance(index_u, index_v)
					inner_cooccur.append(cooccurance)
					inner_dist.append(jac_dist)
		dists.append(reduce(lambda x,y: x+y, inner_dist)/len(inner_dist))
		cooccurs.append(reduce(lambda x,y: x+y, inner_cooccur)/len(inner_cooccur))
		#print "the average jaccard distance in cluster %s is %f" %(key,dists);
		#print "the average cooccurance in cluster %s is %d" %(key,cooccurs);
	avg_dist = reduce(lambda x,y: x+y, dists)/len(dists)
	avg_cooccurs = reduce(lambda x,y: x+y, cooccurs)/len(cooccurs)
	return avg_dist,avg_cooccurs
	# print 'the average jaccard distance is', avg_dist
	# print 'the average cooccurance is', avg_cooccurs

def get_vec_from_model(filename,size):
	#size=parameter
	vec_data = np.genfromtxt(filename,
	#vec_data = np.genfromtxt(model_file,
		delimiter=' ',
		dtype=None,
		usecols=(_ for _ in range (1,size+1)),
		autostrip=True,skip_header=2)
	ingr_names = np.genfromtxt(filename,
		delimiter=' ',
		dtype=None, #major ingredients
		usecols=0,
		autostrip=True,skip_header=2)
	global vol_num
	vol_num = len(vec_data)
	#print 'size=',size,'volcabulary:',vol_num
	#X = StandardScaler().fit_transform(vec_data)
	return vec_data,ingr_names

if __name__ == "__main__":
	main()