import multiprocessing
from time import time

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import max_weight_matching

from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph

model_folder = '../vectors/word2vec_vec_gegativeseted/'
#model_file = '../tmp/model_sg_hs_200_13_1e-4'
n_clusters = 14
size = 200
def get_vec(negative,window):
	vec_data = np.genfromtxt(model_folder+'model_'+str(negative)+'_'+str(window),
	#vec_data = np.genfromtxt(model_file,
		delimiter=' ',
    	dtype=None,
    	usecols=(_ for _ in range (1,size+1)),
    	autostrip=True,skip_header=2)
	#X = StandardScaler().fit_transform(vec_data)
	return vec_data

#affinity propogation
def affinity_cluster(negative,window):
	score = 0
	X = get_vec(negative=negative,window=window)
	dampings = [0.5,0.6,0.7,0.8,0.9]
	for damping in dampings:
		for preference in range(-200,-50,10):
			#print damping,preference
			ap = cluster.AffinityPropagation(damping=damping,preference=preference)
	        pred = ap.fit(X).labels_
	        tmp = calc_score(pred)
	        if tmp > score :
	        	score = tmp
	        	best_damping = damping
	        	best_preference = preference
	# with open('../results/affinity_cluster_1.res','a') as f:
	# 	f.writelines("%d %d %f %d\n" %(negative,window,score,best_damping,best_preference))
	# f.close()
	print negative,window,score,best_damping,best_preference

#agglomerative clustering
def agglomerative_cluster(size,negative,window):
	score = 0
	X = get_vec(size=size, negative=negative,window=window)
	linkages = ['complete', 'average']
	affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine','precomputed']
	for i in range(1,14):
		for linkage in linkages:
			for affinity in affinities:
				connectivity = kneighbors_graph(X, n_neighbors=i, include_self=False)
				connectivity = 0.5 * (connectivity + connectivity.T)
				agglo = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine',
					linkage='average',connectivity=connectivity)
				pred = agglo.fit(X).labels_
	        	tmp = calc_score(pred)
	        	if tmp > score :
		        	score = tmp
		        	best_linkage = linkage
		        	best_affinity = affinity
		        	best_neighber = i
	with open('../results/agglomerative_cluster.res','a') as f:
		f.writelines("%d %d %d %s %s %d\n" %(size,negative,window,score,best_linkage,best_affinity,i))
		#f.writelines(size+' '+negative+' '+score+' '+best_linkage+' '+best_affinity+' '+i+'\n')
	f.close()
	print size,negative,score,best_linkage,best_affinity,i

def birch_cluster(negative,window):
	score = 0
	X = get_vec(negative=negative,window=window)
	for threshold in np.arange(0.3,1.0,0.1):
		for branching_factor in range(40,100,10):
			birch = cluster.Birch(threshold=threshold,branching_factor=branching_factor,
				n_clusters=n_clusters)
			pred = birch.fit(X).labels_
	        tmp = calc_score(pred)
	        if tmp > score :
	        	score = tmp
	        	best_threshold = threshold
	        	best_factor = branching_factor
	# with open('../results/birch_cluster.res','a') as f:
	# 	f.writelines("%d %d %d %f %d\n" %(size,negative,window,score,best_threshold,best_factor))
	# 	#f.writelines(size+' '+negative+' '+score+' '+best_linkage+' '+best_affinity+' '+i+'\n')
	# f.close()
	print negative,window,score,best_threshold,best_factor

def dbscan_cluster(size,negative,window):
	score = 0
	X = get_vec(size=size, negative=negative,window=window)
	algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
	for eps in np.arange(0.1,0.6,0.1):
		for algorithm in algorithms:
			dbscan = cluster.DBSCAN(eps=eps,algorithm=algorithm)
			pred = dbscan.fit(X).labels_
        	tmp = calc_score(pred)
        	if tmp > score :
	        	score = tmp
	        	best_eps = eps
	        	best_algorithm = algorithm
	with open('../results/dbscan_cluster.res','a') as f:
		f.writelines("%d %d %d %f %s\n" %(size,negative,window,score,best_eps,best_algorithm))
		#f.writelines(size+' '+negative+' '+score+' '+best_linkage+' '+best_affinity+' '+i+'\n')
	f.close()
	print size,negative,score,best_threshold,best_factor	

#kmeans
def kmeans_cluster(size,negative,window):
	score = 0
	X = get_vec(size=size, negative=negative,window=window)
	for i in range(10,31,10):
		k_means = cluster.KMeans(n_clusters=n_clusters,n_init=i)
		pred = k_means.fit(X).labels_
		tmp = calc_score(pred)
    	if tmp > score :
        	score = tmp
        	best_init = i
	with open('../results/kmeans_cluster.res','a') as f:
		f.writelines("%d %d %d %d\n" %(size,negative,window,score,best_init))
		#f.writelines(size+' '+negative+' '+score+' '+best_linkage+' '+best_affinity+' '+i+'\n')
	f.close()
	print size,negative,score,best_init

#meanshit
def ms_cluster(size,negative,window):
	score = 0
	X = get_vec(size=size, negative=negative,window=window)
	#ms = cluster.MeanShift(bandwidth=n_clusters, bin_seeding=True)
	#pred = ms.fit(X).labels_
	#score = calc_score(pred)
	#print 'defined cluster 14, score:',score
	#for quantile in np.arange(0.1,1.0,0.1): #np.linspace(0.1,1,10)
		#print quantile
	#bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
	ms = cluster.MeanShift(bandwidth=14, bin_seeding=True)
	pred = ms.fit(X).labels_
	tmp = calc_score(pred)
	#print tmp
	if tmp > score :
		score = tmp
    	#best_quantile = quantile
        	#print 'best quantile:',best_quantile
	print 'for vector size',size,',negative number',negative,',<MeanShift> best score is',score

def spectual_cluster(size,negative,window):
	score = 0
	X = get_vec(size=size, negative=negative,window=window)
	spectral = cluster.SpectralClustering(n_clusters=n_clusters,
		eigen_solver='arpack',affinity="nearest_neighbors")

def calc_score(pred):
	B = nx.Graph()
	#generate bipartite graph 0:
	for item in np.nditer(uniq):
		name = str(item)
		exec("%s = nx.Graph(name='%s')" % (name,name))
		exec("B.add_node(%s,bipartite=0)" % (name))
	for i in range(len(ingr_all)):
		graph = str(ingr_all[i,1]).replace('/','_')
		node = str(ingr_all[i,0])
		exec("%s.add_node('%s')" % (graph,node))
	#print ingr_names.shape,'and',pred.shape
	#print pred
	labels = np.row_stack((ingr_names,pred))	
	pred_uniq = np.unique(labels[1])
	#print len(pred_uniq)
	#generate bipartite graph 0:
	for item in np.nditer(pred_uniq):
		name = 'c_'+str(item)
		exec("%s = nx.Graph(name='%s')" % (name,name))
		exec("B.add_node(%s,bipartite=1)" % (name))
		for graph in list(uniq):
			exec("B.add_edge(%s,%s,weight=0)" % (graph,name))

	for i in range(len(labels[0])):
		graph = 'c_'+str(labels[1,i])
		node = str(labels[0,i])
		exec("%s.add_node('%s')" % (graph,node))

	for graph in list(uniq):
		exec("graph_0 = %s" %(graph))
		for nodes in graph_0:
			for graph_ in list(pred_uniq):
				name = 'c_'+str(graph_)
				exec("graph_1 = %s" %(name))
				#name='c_'+str(graph_)
				#print graph_0.nodes()
				if nodes in graph_1.nodes():
					B[graph_0][graph_1]['weight'] += 1
	score=0
	for top, bottom in max_weight_matching(B).iteritems():
	 	score += B[top][bottom]['weight']
	return (score/2)

ingr_names = np.genfromtxt(model_folder+'model_15_16',delimiter=' ',
	dtype=None, #199 major ingredients
	usecols=0,
	autostrip=True,skip_header=2)

ingr_all = np.genfromtxt('../data/ingredients.csv',delimiter=',',
	dtype=None,  #381 ingredients in total
	usecols=(0,1),
	autostrip=True,skip_header=0)
#print np.array2string(ingr_all)

#uniq is all the food categories
uniq = np.unique(ingr_all[:,1])
for i in range(len(uniq)):
	uniq[i] = str(uniq[i]).replace('/', '_')

for window in range(3,17):
	#affinity_cluster(negative=15,window=window)
	#agglomerative_cluster(size=size,negative=negative)
	birch_cluster(negative=15,window=window)
	#ms_cluster(size=size,negative=negative)
	#kmeans_cluster(size=size,negative=negative)
