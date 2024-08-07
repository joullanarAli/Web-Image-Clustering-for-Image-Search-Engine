from abc import ABC, abstractmethod
class ClusteringInterface(ABC):
    
    @abstractmethod
    def fit(self, data):
        """Fit the clustering model to the data."""
        pass
    
    @abstractmethod
    def predict(self, data):
        """Predict the clusters for the data."""
        pass
    
    @abstractmethod
    def fit_predict(self, data):
        """Fit the model to the data and predict clusters."""
        pass
     
    @abstractmethod
    def save_clustered_images(self, data):
        """saving clustered images."""
        pass
    
