import multiprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import max_weight_matching

# from sklearn.preprocessing import StandardScaler
from sklearn import cluster
# from sklearn.neighbors import kneighbors_graph

size = 35
#n_clusters = 9
vol_num = 0

ingr_all = np.genfromtxt('../data/ingredients.csv',delimiter=',',
	dtype=None,  #381 ingredients
	usecols=(0,1),
	autostrip=True,skip_header=0)
#uniq is all the food categories
uniq = np.unique(ingr_all[:,1])
for i in range(len(uniq)):
	uniq[i] = str(uniq[i]).replace('/', '_')

co_occur_matrix = np.genfromtxt('../data/co_occur_matrix.txt',delimiter=' ',
	#names=names,
	dtype=int,
	usecols=np.arange(0,381),
	autostrip=True,
	skip_header=0)

def main():
	parameters = list(range(3,14))
	match_scores = list()
	model_file = './model_2'
	X,ingredients = get_vec_from_model(filename=model_file)
	for n_clusters in range(3,14):				
		model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
			linkage='average',affinity="cosine")
		pred = model.fit(X).labels_
		y = calc_match_score(pred,ingredients)
		match_scores.append(y)

	print match_scores

	plt.plot(parameters,match_scores)
	plt.xlabel('cluster number')
	plt.ylabel('score')

	plt.xlim(3,14)
	plt.xticks(np.linspace(3,14,12, endpoint=True))

	plt.ylim(0,100)
	plt.yticks(np.linspace(0,100,11, endpoint=True))
	plt.title('score variance')
	plt.savefig('./figure_CompareCluster.png')
	#plt.show()

def get_vec_from_model(filename):
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

def calc_match_score(pred,ingredients):
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
	#print ingredients.shape,'and',pred.shape
	#print pred
	labels = np.row_stack((ingredients,pred))
	pred_uniq = np.unique(labels[1])
	#print len(pred_uniq)

	#generate bipartite graph 1:
	for item in np.nditer(pred_uniq):
		name = 'c_'+str(item)
		#print name
		exec("%s = nx.Graph(name='%s')" % (name,name))
		exec("B.add_node(%s,bipartite=1)" % (name))
		for graph in list(uniq):
			exec("B.add_edge(%s,%s,weight=0)" % (graph,name))
	# for i in range(len(B.nodes())):
	# 	print B.nodes()[i]

	#print list(labels[1])
	#generate bipartite graph 1:
	for i in range(len(labels[0])):
		graph = 'c_'+str(labels[1,i])
		node = str(labels[0,i])
		exec("%s.add_node('%s')" % (graph,node))

	for graph in list(uniq):
		exec("graph_0 = %s" %(graph)) #graph_0 is the food category that are being matched now
		for nodes in graph_0: #nodes represent the original food categories
			for graph_ in list(pred_uniq):
				name = 'c_'+str(graph_)
				exec("graph_1 = %s" %(name)) #name represents the cluster labels c_i
				#name='c_'+str(graph_)
				#print graph_0.nodes()
				if nodes in graph_1.nodes():
					B[graph_0][graph_1]['weight'] += 1

				#exec("if nodes in %s.nodes():\n\tprint %s" %(name,name))
				#exec("if nodes in %s.nodes():\n	B[%s][%s]['weight'] += 1" %(name,graph,name))
				#if nodes in graph_.nodes():
	match_score = 0
	cooccur_score = 0
	matching = max_weight_matching(B,maxcardinality=True)
	for key in matching.keys():
		if str(key).startswith('c_'): #keep uniq match result
			matching.pop(key)

	for top, bottom in matching.iteritems():
		match_score += B[top][bottom]['weight']
		#print 'match_score:',(match_score/float(vol_num))

	  	correct = list()
	 	co_occur = list()
	 	for nodes in top.nodes():
	 		if nodes in bottom.nodes():
	 			correct.append(nodes)
	 	#print correct
	 	for item in correct:
	 		index = list(ingr_all[:,0]).index(item)
	 		top = np.argsort( co_occur_matrix[index] )[379:374:-1]
	 		co_occur.append( ingr_all[top,0] )
	 		#print 'co_occur:',co_occur
	 	uniq_occur = np.unique(co_occur)
	 	for nodes in bottom.nodes():
	 		if ((nodes not in top) and (nodes in uniq_occur)):
	 			cooccur_score += 1 #cooccur_score =
	#print 'cooccur_score:',cooccur_score
	#print 'match_score:',match_score
	return (match_score+cooccur_score/float(vol_num))

if __name__ == "__main__":
    main()