import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import max_weight_matching
import os
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict
import imp
import sys
sys.path.append('../vec_util/')
from dist_calc import calculate_jaccard,calculate_cooccurance
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

size = 5
vol_num = 0
#n_clusters = 2
#clusters_max = 14

def main():
	#parameters = list(range(1,30))
	#match_scores = list()
	best_scores = np.zeros((30,2))
	for negative in range(1,31):
		model_file = './model_'+str(negative)
		#model_file = './model_'
		X,ingredients = get_vec_from_model(filename=model_file,size=size)
		labels=list(ingredients)
		
		# distance_matrix = cosine_distances(X)
		# distance_array = pdist(X,metric='cosine')
		# print distance_matrix.shape
		# print distance_array.shape
		#print np.array2string(distance_matrix[0:2,:])

		#print similarity_matrix.shape
		#Z = linkage(X,method='average' ,metric='cosine')
		Z = linkage(X,method='average',metric='cosine')
		#print Z
		#save_dendrogram(parameter=5,Z=Z,labels=labels)
		#for t in np.arange(0.2,0.41,0.02):
		scores = np.zeros((12,2))#,dtype=[('n_cluster','i4'),('score','f8')])
		for n_clusters in range(3,15):
			cluster_indexes = fcluster(Z,n_clusters,criterion='maxclust')
			#print cluster_indexes
			score = silhouette_score(X, cluster_indexes,metric='cosine')
			scores[n_clusters-3,0]=n_clusters
			scores[n_clusters-3,1]=score

			# samples = silhouette_samples(X, cluster_indexes,metric='cosine')
			#print 't=',t,':',score
			#print 'k=',k,':',score
		# print np.array2string(samples)
		#print np.array2string(scores)
		best = scores[np.argmax(scores[:,1]),:]
		best_scores[negative-1,:]=best
		#best_scores[size/5,1]=best[1]
		#print np.array2string(best)
		print best_scores[negative-1,:]
		#break
	print 'best negative sample:',np.argmax(best_scores[:,1])+1
		#break


def save_dendrogram(parameter,Z,labels):
	plt.figure(figsize=(20, 50))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.ylabel('sample index')
	plt.xlabel('distance')
	dendrogram(
		Z,
		#truncate_mode='lastp',
		#p=12,
		labels=labels,
		orientation='left',
		#leaf_rotation=90.,  # rotates the x axis labels
		leaf_font_size=8.,  # font size for the x axis labels
	)
	for i in np.arange(0.4,1.0,0.1):
		plt.axvline(x=i, c='k',linestyle='dashed')
	#plt.show()
	plt.savefig('./gensim_test_'+str(parameter)+'_labeled.png')

def plot_knee_point():
	fig, axes23 = plt.subplots(2, 3)
	for method, axes in zip(['single', 'complete'], axes23):
		z = linkage(X,method=method,metric='cosine')#.astype(int)
		#save_dendrogram(parameter=negative,Z=Z)
		#print np.array2string(z)
		#break
		# Plotting
		axes[0].plot(range(1, len(z)+1), z[::-1, 2])
		knee = np.diff(z[::-1, 2], 2)
		axes[0].plot(range(2, len(z)), knee)

		num_clust1 = knee.argmax() + 2
		knee[knee.argmax()] = 0
		num_clust2 = knee.argmax() + 2
		knee[knee.argmax()] = 0
		num_clust3 = knee.argmax() + 2

		axes[0].text(num_clust1, z[::-1, 2][num_clust1-1], 'possible\n<- knee point')

		part1 = fcluster(z, num_clust1, 'maxclust')
		part2 = fcluster(z, num_clust2, 'maxclust')
		part3 = fcluster(z, num_clust2, 'maxclust')
		m = '\n(method: {})'.format(method)
		plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
			ylabel='{}\ncluster distance'.format(m))
		#plt.setp(axes[1], title='{} Clusters'.format(num_clust1))
		#plt.setp(axes[2], title='{} Clusters'.format(num_clust2))

	plt.tight_layout()
	plt.show()

def plot_info_gain():
    K_MAX = 15
    KK = range(1,K_MAX+1)

    KM = [kmeans(X,k) for k in KK]
    centroids = [cent for (cent,var) in KM]
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
    betweenss = totss - tot_withinss          # The between-cluster sum of squares

    ##### plots #####
    #kIdx = 9        # K=10
    clr = cm.spectral( np.linspace(0,1,10) ).tolist()
    mrk = 'os^p<dvh8>+x.'

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK, betweenss/totss*100, 'b*-')
    # ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, 
    #     markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    ax.set_ylim((0,100))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    plt.show()

def calc_dist():
	# cluster_indexes = fcluster(Z,t=0.7,criterion='distance')
	#indexes = np.argsort(clusters)
	#print len(cluster_indexes)

	#d: a dictionary using a cluster index as key, and a list of ingredients in this cluster as value
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
	print 'the average jaccard distance is', avg_dist
	print 'the average cooccurance is', avg_cooccurs

def graph_variance_by_parameter(parameter):
	pass




		# for mode in ('connectivity','distance'):
		# 	knn_graph = kneighbors_graph(X, n_clusters,mode=mode)
		# 	model = cluster.AgglomerativeClustering(n_clusters=n_clusters,connectivity=knn_graph,
		# 		linkage='average',affinity="cosine",compute_full_tree=True)
		# 	children = model.fit(X).children_
		# 	print np.array2string(np.sort(children,axis=0))
		# 	print model.fit(X).n_components_
		# 	#drawdendrogram(children,X,jpeg='test.jpg')
		# 	break
		#break
		# pred = model.fit(X).labels_
		# y = calc_match_score(pred,ingredients)
		# match_scores.append(y)

	# print match_scores

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