from sklearn.cluster import KMeans
from IClustering import ClusteringInterface

class KMeansClustering(ClusteringInterface):
    
    def __init__(self, n_clusters=5):
        self.model = KMeans(n_clusters=n_clusters)
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def fit_predict(self, data):
        return self.model.fit_predict(data)
