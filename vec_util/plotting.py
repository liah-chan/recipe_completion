import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import math
from scipy.cluster.hierarchy import dendrogram

average_similarity = 1.23594857446

def plot_similarity(parameters,similarities,para_name):
	plt.plot(parameters,similarities)
	plt.xlabel(para_name)
	plt.ylabel('average similarity')

	plt.xlim(parameters[0],parameters[-1])
	plt.xticks(np.linspace(parameters[0],parameters[-1],len(parameters), endpoint=True))

	plt.ylim(0,5)
	plt.yticks(np.linspace(0,5,6, endpoint=True))
	plt.title('average similarity varying '+para_name)

	maximum = max(similarities)
	index = similarities.index(maximum)
	plt.annotate(maximum,(parameters[index],similarities[index]))
	plt.axvline(parameters[index],color='b',linestyle='--')

	plt.axhline(average_similarity,color='r')
	plt.text(parameters[0], math.floor(average_similarity), 'average:'+str(average_similarity))

	plt.savefig('./figure_compare_'+para_name+'.png')
	# plt.show()

def plot_similarity_distribution(similarity_matrix_file,metric,upper_lim,lower_lim,interval,max_value,min_value,avg_value):
	similarity_matrix = np.genfromtxt(similarity_matrix_file,
			delimiter=' ',
			dtype=float,
			#usecols=(0),
			autostrip=True,skip_header=0)
	n = int((upper_lim - lower_lim)/interval) 
	# print n
	y=[0]*n

	if lower_lim < 0:

		for i in range(0,248):
			for j in range(i+1,248):
				similarity = int(math.floor(similarity_matrix[i,j]/interval))+25 #(-25,25)
				y[similarity]+=1
		# print y[0:50] 

	else:

		for i in range(0,248):
			for j in range(i+1,248):
				similarity = int(math.floor(similarity_matrix[i,j]/interval))
				y[similarity]+=1

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.bar(np.arange(lower_lim,upper_lim,interval), y,width=interval,color='w')

	ax.set_xlabel('Similarity')
	ax.set_ylabel('Pair Number')
	ax.set_xlim(lower_lim,upper_lim,interval)
	ax.set_ylim(0,2500,500)
	ax.grid(True)
	ax.axvline(avg_value,color='y')
	ax.text(avg_value,250,'avg:'+str(avg_value))
	ax.axvline(max_value,color='g')
	ax.text(max_value,250,'max:'+str(max_value))
	ax.axvline(min_value,color='g')
	ax.text(min_value,250,'min:'+str(min_value))
	# ax.text(1.3,2250,'average:'+str(average_similarity))
	plt.savefig(metric+'_similarity_distribution.png')
	# plt.show()

def plot_similarity_by_category(categories,similarities):
	# y = [1.27451979788,1.283164108,1.81809752843,1.88457866024,1.47456968967,1.04514668201,0.955649381568,1.48785275225,1.59540143708,1.49063714275,1.23826081113,1.20626177708,1.60856022653]
	# x = ['plant_derivative','plant','meat','herb','animal_product','alcoholic_beverage','fruit','dairy','fish/seafood','vegetable','nut/seed/pulse','cereal/crop','spice']
	x = range(0,len(similarities)*3,3)
	y = similarities
	plt.bar(x, y, width=3,color='w')
	plt.xlabel('food category')
	plt.ylabel('similarity')
	plt.ylim(0,2,0.5)
	# plt.yticks(np.linspace(0,2,0.5,endpoint=True))
	# plt.xticks(range(0,len(y)*3,3),['plant_derivative','plant','meat','herb','animal_product','alcoholic_beverage','fruit','dairy','fish/seafood','vegetable','nut/seed/pulse','cereal/crop','spice'],fontsize=8,rotation=45)
	plt.xticks(range(0,len(y)*3,3),categories,fontsize=8,rotation=45)
	plt.axhline(average_similarity,color='r')
	plt.text(0, 1.3, 'average'+str(average_similarity))
	plt.savefig('similarity_by_category.png')

def save_dendrogram(parameter,Z,labels,n_clusters):
	plt.figure(figsize=(20, 50))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.ylabel('sample index')
	plt.xlabel('distance')
	dendrogram(
		Z,
		# truncate_mode='lastp',
		# p=n_clusters,
		labels=labels,
		orientation='left',
		#leaf_rotation=90.,  # rotates the x axis labels
		leaf_font_size=8.,  # font size for the x axis labels
	)
	for i in np.arange(0.4,1.0,0.1):
		plt.axvline(x=i, c='k',linestyle='dashed')
	#plt.show()
	plt.savefig('./Dendrogram_'+str(parameter)+'_labeled.png')

def main():
	# plot_similarity_by_category()
	plot_similarity_distribution()

if __name__ == "__main__":
    main()