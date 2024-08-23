from fastapi import FastAPI,HTTPException
from typing import List, Dict
from RetrievalAndClusteringSystem.DatasetReader.FlickrDataset import FlickrDataset_reader
from RetrievalAndClusteringSystem.ModelsUsage.ModelReader.CLIP_reader import CLIP_reader
from fastapi.responses import HTMLResponse
from RetrievalAndClusteringSystem.RetrievalSystem.Faiss_Sen_Retrieval import Faiss_Sen_Retrieval
from RetrievalAndClusteringSystem.RetrievalSystem.Fiss_CLIP_Retrieval import Faiss_CLIP_Retrieval
from RetrievalAndClusteringSystem.Clustering.FaissKMeansClustering import FaissKMeansClustering
from RetrievalAndClusteringSystem.RetrievalSystem.my_retrieval import My_Retrieval
import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles
import logging
from fastapi.responses import JSONResponse
import os
import torch
from fastapi.middleware.cors import CORSMiddleware
import sys

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.get('/search')
def search_images(query: str):
    print('hello')
    clusters = get_similar_images(query)
    return {"query": query, "clusters": clusters}

@app.get("/cluster")
async def get_cluster(cluster: str):
    cluster_dir = os.path.join("static", "clusters", f"cluster_{cluster}")
    if not os.path.isdir(cluster_dir):
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    images = [img for img in os.listdir(cluster_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    return JSONResponse({"images": images})






# def get_similar_images(query: str):
#     dataset = FlickrDataset_reader()
#     df, image_paths, captions = dataset.readDataset()

#     distance_metrice = 'cos_similarity'
#     sen_system = Faiss_Sen_Retrieval(distance_metrice)
#     dataset_folder = 'flickr'
#     faiss_sen_index, similarities, indices, retrieved_embeddings = sen_system.search(query, image_paths, captions, dataset_folder)

#     # Get clip embeddings
#     clip_model = CLIP_reader()
#     processor, model = clip_model.readModel()
#     clip_embeddings = torch.load('image_embeddings.pt').numpy()  # Adjust this as necessary

#     # Reshape and align embeddings
#     if len(retrieved_embeddings.shape) == 3:
#         retrieved_embeddings = retrieved_embeddings.squeeze(0)

#     # Ensure embeddings are aligned
#     if clip_embeddings.shape[1] != retrieved_embeddings.shape[1]:
#         raise ValueError("The dimensions of embeddings do not match.")

#     # Combine embeddings
#     combined_embeddings = np.vstack((retrieved_embeddings, clip_embeddings))

#     # Continue with clustering and other processing
#     n_clusters = 7
#     clusters = {}

#     embedding_dim = combined_embeddings.shape[1]
#     # clustering_model = FaissKMeansClustering(d=embedding_dim, n_clusters=n_clusters)
#     # cluster_centers, labels = clustering_model.fit(combined_embeddings)
#     # clustering_model.cluster_and_save_images(combined_embeddings, image_paths, indices)

#     # # Convert numpy.int64 labels to int
#     # labels = [int(label) for label in labels]

#     # for label, image_path in zip(labels, images):
#     #     if label not in clusters:
#     #         clusters[label] = []
#     #     clusters[label].append(image_path)
#     # return clusters

@app.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint"}
def get_similar_images(query: str):
    dataset = FlickrDataset_reader()
    df, image_paths, captions = dataset.readDataset()
    distance_metrice = 'cos_similarity'
    my_retrieval = My_Retrieval(distance_metrice)
    k=500
    alpha = 0.5
    n_clusters = 2
    #my_retrieval.retrieveAndCluster(image_paths,captions,query,k,alpha,n_clusters)
    cluster_centers, labels, top_images = my_retrieval.retrieveAndCluster(image_paths,captions,query,k,alpha,n_clusters)
    clusters = {}
    for label, image_path in zip(labels, top_images):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image_path)
    return clusters


# def get_similar_images(query: str):
#     dataset = FlickrDataset_reader()
#     df, image_paths, captions = dataset.readDataset()

#     distance_metrice = 'cos_similarity'
#     sen_system = Faiss_Sen_Retrieval(distance_metrice)
#     dataset_folder = 'flickr'
#     k=500
#     faiss_sen_index, similarities, indices, retrieved_embeddings = sen_system.search(query, image_paths, captions, k)

#     samples = {
#         "caption": [captions[i] for i in indices[0]],
#         "image": [image_paths[i] for i in indices[0]],
#         "similarities": similarities[0].tolist(),
#     }

#     images = []
#     samples_df = pd.DataFrame.from_dict(samples)
#     samples_df["similarities"] = similarities[0]
#     samples_df.sort_values("similarities", ascending=False, inplace=True)
#     for _, row in samples_df.iterrows():
#         images.append(row.image)

    
#     n_clusters = 7  
#     clusters = {}

#     # Ensure retrieved_embeddings is a 2D array
#     if len(retrieved_embeddings.shape) == 3:
#         retrieved_embeddings = retrieved_embeddings.reshape(-1, retrieved_embeddings.shape[-1])

#     embedding_dim = retrieved_embeddings.shape[1]
#     # Initialize and fit the Faiss K-means clustering model
#     clustering_model = FaissKMeansClustering(d=embedding_dim, n_clusters=n_clusters)
#     cluster_centers, labels = clustering_model.fit(retrieved_embeddings)
#     clustering_model.cluster_and_save_images(retrieved_embeddings, images, indices)

#     # Convert numpy.int64 labels to int
#     labels = [int(label) for label in labels]

#     for label, image_path in zip(labels, images):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(image_path)
#     return clusters



