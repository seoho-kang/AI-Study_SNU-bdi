import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

from numpy.random import randint
from PIL import Image

np.random.seed(5)

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

img = Image.open("photo.jpg")
img = np.array(img, dtype=np.float64)
w, h, d = original_shape = tuple(img.shape)
X = np.reshape(img, (w * h, d))

estimators=[('k=8', KMeans(n_clusters=8))]

fignum = 1
for name, est in estimators:
	fig = plt.figure(fignum, figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	km=est.fit(X)
	labels = est.labels_

	print "Silhouette score(%s): %f"%(name, metrics.silhouette_score(X, est.labels_, sample_size=1000))
			
 	ax.scatter(X[:, 0], X[:, 1],X[:, 2],c=labels.astype(np.float), edgecolor='k')
 
 	ax.w_xaxis.set_ticklabels([])
 	ax.w_yaxis.set_ticklabels([])
 	ax.w_zaxis.set_ticklabels([])
 	ax.set_xlabel('Red')
 	ax.set_ylabel('Green')
 	ax.set_zlabel('Blue')
 	ax.set_title(name)
 	ax.dist = 12
 	fignum = fignum + 1

	plt.show()

