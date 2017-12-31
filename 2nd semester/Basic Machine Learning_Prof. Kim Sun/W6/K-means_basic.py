from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print "Clusters %s"%(kmeans.labels_)
print "Cluster centroids: %s"%(kmeans.cluster_centers_)
print "Prediction cluster of [0, 0], [4, 4]: %s"%(kmeans.predict([[0, 0], [4, 4]]))

