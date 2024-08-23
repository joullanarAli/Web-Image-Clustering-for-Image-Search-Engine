from RetrievalAndClusteringSystem.Clustering.IClustering import ClusteringInterface
from RetrievalAndClusteringSystem.Clustering.GenericClustering import GenericClustering
import faiss
import os
from PIL import Image
import numpy as np
import shutil

from sklearn.model_selection import ParameterGrid

class FaissKMeansClustering(ClusteringInterface,GenericClustering):
    
    def __init__(self,d, n_clusters, niter=20):
        self.d = d
        self.n_clusters = n_clusters
        self.model = faiss.Kmeans(d=d, k=n_clusters, niter=niter, verbose=True)
    
    def fit(self, data):
        data = np.array(data, dtype=np.float32)
        self.model.train(data)
        cluster_centers = self.model.centroids
        _, labels = self.model.index.search(data, 1)
        labels = labels.flatten()
        self.labels = labels
        self.cluster_centers = cluster_centers
        return cluster_centers, labels
    
    def predict(self, data):
        return self.model.predict(data)
    
    def fit_predict(self, data):
        return self.model.fit_predict(data)
    
    def print_clusters(self):
        print("faiss_K-means Clusters:", self.labels)
        print("faiss_K-means Cluster centers:", self.cluster_centers)

    def cluster_and_save_images(self, embeddings, image_paths, indices, root_folder='static\\clusters'):
        # Ensure embeddings is a 2D array
        if len(embeddings.shape) == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        embedding_dim = embeddings.shape[1]

        # Fit the Faiss K-means clustering model
        cluster_centers, labels = self.fit(embeddings)

        # Create folders for each cluster
        os.makedirs(root_folder, exist_ok=True)

        self.save_clustered_images(image_paths, root_folder)
        # for i in range(self.n_clusters):
        #     os.makedirs(os.path.join(root_folder, f'cluster_{i}'), exist_ok=True)

        # # Save images to the corresponding folders
        # for idx, cluster in enumerate(labels):
        #     image_path = os.path.join('.\\Dataset\\FlickrDataset\\Images', image_paths[idx])
        #     image = Image.open(image_path)

        #     save_path = os.path.join(root_folder, f'cluster_{cluster}', os.path.basename(image_path))
        #     image.save(save_path)
        #     print(f"Image {image_path} saved to {save_path}")

        # print("Images have been clustered and saved.")


    def save_clustered_images(self, image_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for cluster_id in range(self.n_clusters):
            cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)

        for idx, cluster_id in enumerate(self.labels):
            try:
                src_image_path = '.\\RetrievalAndClusteringSystem\\Dataset\\FlickrDataset\\Images\\'+image_paths[idx]
                cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
                dst_image_path = os.path.join(cluster_dir, os.path.basename(src_image_path))
                shutil.copy(src_image_path, dst_image_path)
                print(f"Image {src_image_path} saved to {dst_image_path}")
            except Exception as e:
                print(f"Failed to save image {src_image_path}: {e}")

        print("Images have been clustered and saved.")


    
    def perform_grid_search(self,param_grid, embeddings, evaluation_metric):
        best_params = None
        best_score = float('-inf') if evaluation_metric == 'silhouette' else float('inf')
        best_labels = None
        best_centroids = None
        
        for params in ParameterGrid(param_grid):
            n_clusters = params['n_clusters']
            niter = params['niter']
            
            d = embeddings.shape[1]  # Dimension of embeddings
            kmeans = FaissKMeansClustering(d=d, n_clusters=n_clusters, niter=niter)
            
            try:
                centroids, labels = self.fit(embeddings)
                
                silhouette, davies_bouldin, calinski_harabasz = kmeans.evaluate_clustering(embeddings, labels)
                
                if evaluation_metric == 'silhouette' and silhouette > best_score:
                    best_score = silhouette
                    best_params = params
                    best_labels = labels
                    best_centroids = centroids
                elif evaluation_metric == 'davies_bouldin' and davies_bouldin < best_score:
                    best_score = davies_bouldin
                    best_params = params
                    best_labels = labels
                    best_centroids = centroids
                elif evaluation_metric == 'calinski_harabasz' and calinski_harabasz > best_score:
                    best_score = calinski_harabasz
                    best_params = params
                    best_labels = labels
                    best_centroids = centroids
                elif evaluation_metric == 'sse':
                    sse = self.calculate_sse(embeddings, labels, centroids)
                    if sse < best_score:
                        best_score = sse
                        best_params = params
                        best_labels = labels
                        best_centroids = centroids
                        
            except Exception as e:
                print(f"An error occurred during KMeans training with params {params}: {e}")
        return best_params, best_score, best_labels, best_centroids


    
        # Assign best labels to the DataFrame
        #samples_df['cluster_label'] = best_labels