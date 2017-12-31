import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics # 실루엣스코어 계산할때 편리하더라
from reader import reader

np.random.seed(5)

data_reader = reader()
X = data_reader.x
y = data_reader.y

estimators = [('k=2', KMeans(n_clusters=2)), 
              ('k=3', KMeans(n_clusters=3)),
              ('k=4', KMeans(n_clusters=4)),
              ('k=5', KMeans(n_clusters=5)),
              ('k=6', KMeans(n_clusters=6)),
              ('k=7', KMeans(n_clusters=7)),
              ('k=8', KMeans(n_clusters=8))]             

# print metric measure headers
print ("Estimator\tHomogeneity\tCompleteness\tV-means\t\tARI\t\tAMI\t\tSilhouette") # ari : 정답set이 있을때만 / Silhouette : 정답set 없을때도 사용가능

fignum = 1
titles = ['2 clusters', '3 clusters','4 clusters','5 clusters','6 clusters','7 clusters','8 clusters']
#titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
	fig = plt.figure(fignum, figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	est.fit(X) # est 하나가 (a)임
	labels = est.labels_

	# measure cluster qualities
	print('%s\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f'
          % (name, 
             metrics.homogeneity_score(y, est.labels_),
             metrics.completeness_score(y, est.labels_),
             metrics.v_measure_score(y, est.labels_),
             metrics.adjusted_rand_score(y, est.labels_),
             metrics.adjusted_mutual_info_score(y, est.labels_),
             metrics.silhouette_score(X, est.labels_, metric='euclidean')))
			

	ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Dimension1') # Petal width
	ax.set_ylabel('Dimension2') # Sepal length
	ax.set_zlabel('Dimension3') # Petal length
	ax.set_title(titles[fignum - 1])
	ax.dist = 12
	fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
	ax.text3D(X[y == label, 3].mean(), X[y == label, 0].mean(), X[y == label, 2].mean() + 2, name, horizontalalignment='center', bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

#fig.show()
plt.show()

