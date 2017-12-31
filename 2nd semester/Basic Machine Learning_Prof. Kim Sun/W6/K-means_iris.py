import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k=8', KMeans(n_clusters=8)),
              ('k=3', KMeans(n_clusters=3)),
              ('k=3 (random init)', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

# print metric measure headers
print ("Estimator\tHomogeneity\tCompleteness\tV-means\tARI\tAMI\tSilhouette")

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
	fig = plt.figure(fignum, figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	est.fit(X)
	labels = est.labels_

	# measure cluster qualities
	print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
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
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
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

