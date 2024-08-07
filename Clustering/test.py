# main.py
import numpy as np
from KMeansClustering import KMeansClustering
from DBSCANClustering import DBSCANClustering
from FaissKMeansClustering import FaissKMeansClustering

# Sample data
data = np.random.rand(100, 2)

# Using K-means clustering
kmeans = KMeansClustering(n_clusters=3)
kmeans_clusters = kmeans.fit_predict(data)
print("K-means Clusters:", kmeans_clusters)

# Using DBSCAN clustering
# dbscan = DBSCANClustering(eps=0.3, min_samples=5)
# dbscan_clusters = dbscan.fit_predict(data)
# print("DBSCAN Clusters:", dbscan_clusters)


d = data.shape[1]
faiss_kmeans = FaissKMeansClustering(d=d, n_clusters=3)
cluster_centers, faiss_kmeans_clusters = faiss_kmeans.fit(data)
print("faiss_K-means Clusters:", faiss_kmeans_clusters)