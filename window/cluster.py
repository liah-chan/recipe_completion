#import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
#from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

size = 35
vol_num = 0

#def draw_subplot(ax):

def main():
	for window in range(3,33):
		model_file = './model_'+str(window)
		X,ingredients = get_vec_from_model(filename=model_file)
		#fig = plt.figure(figsize=(48,36))	
		scores = np.zeros((11,2))
		for n_clusters in range(3,14):
			model = AgglomerativeClustering(n_clusters=n_clusters,
				linkage='average',affinity="cosine")
			pred = model.fit(X).labels_
			avg_distance=np.zeros((n_clusters,n_clusters))			
			#ax = fig.add_subplot(3,4,n_clusters-2)
			for i in range(n_clusters):
				for j in range(n_clusters):
					# avg_distance[i,j] = pairwise_distances(X[np.where(pred==i)],
					# 	X[np.where(pred==j)],metric="cosine").mean()
					avg_distance[i,j] = cosine_similarity(X[np.where(pred==i)],
						X[np.where(pred==j)]).mean()
			avg_distance /= avg_distance.max()
			diagnal = 0
			off_diagnal = 0
			for i in range(n_clusters):
				for j in range(n_clusters):
					if i==j:
						diagnal += avg_distance[i,j]
					else:
						off_diagnal += avg_distance[i,j]
			# 		ax.text(i, j, '%3.2f' % avg_distance[i,j],
			# 			verticalalignment='center',
			# 			horizontalalignment='center')
			# im = ax.imshow(avg_distance,interpolation='nearest',cmap=plt.cm.gnuplot2,
			# 	vmin=0)
			# ax.set_xticks(range(n_clusters))
			# ax.set_yticks(range(n_clusters))
			score=(off_diagnal/float(n_clusters-1))/float(diagnal)
			#print 'for',n_clusters,'clusters,','c score:',score
			scores[n_clusters-3,:] = [n_clusters,score]
		#print 'scores:',np.array2string(scores)
		#print 'size:',size,';best score:',np.array2string(scores[np.argmin(scores[:,1])] )
		print 'window:',window,';mean score:',np.array2string(np.mean(scores[:,1]))
			#plt.set_title("Interclass cosine distances", size=18)
			#plt.tight_layout()
			#plt.savefig('./'+str(n_clusters)+'_result.png')
		# fig.subplots_adjust(right=0.8)
		# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# fig.colorbar(im, cax=cbar_ax)
		# fig.savefig('./result.png')

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



if __name__ == "__main__":
    main()