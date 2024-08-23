from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from abc import ABC, abstractmethod
import numpy as np

class GenericClustering(ABC):


    def __init__():
        pass


    def calculate_sse(self,embeddings, labels, centroids):
        sse = 0
        for i in range(len(embeddings)):
            sse += np.sum((embeddings[i] - centroids[labels[i]]) ** 2)
        return sse

    def evaluate_clustering(self,embeddings, labels):
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        return silhouette, davies_bouldin, calinski_harabasz
    
    @abstractmethod
    def perform_grid_search(self,param_grid, embeddings, evaluation_metric):
        pass

    def print_evaluation_results(self,metric,param_grid,retrieved_embeddings):
        # Perform grid search
        evaluation_metric = metric  
        best_params, best_score, best_labels, best_centroids = self.perform_grid_search(param_grid, retrieved_embeddings, evaluation_metric)

        print(f"Best Parameters: {best_params}")
        print(f"Best {evaluation_metric.capitalize()} Score: {best_score}")
        return best_labels