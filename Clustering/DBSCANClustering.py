from sklearn.cluster import DBSCAN
from IClustering import ClusteringInterface

class DBSCANClustering(ClusteringInterface):
    
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        # DBSCAN doesn't have a separate predict method
        raise NotImplementedError("DBSCAN does not support predicting new data.")
    
    def fit_predict(self, data):
        return self.model.fit_predict(data)