from RetrievalSystem.IRetrieval import IRetrieval
from Indexing.faiss_indexer import faiss_indexer
from DataPreprocessing.Preprocess import PreprocessData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib
import faiss
import numpy as np

class Faiss_TFIDF_Retrieval(IRetrieval):

    def __init__(self,distance_metrice):
        self.distance_metrice = distance_metrice
        self.embedder = 'TF-IDF'

    
    def search(self, query,image_paths,captions,dataset_folder):
        normalized_tfidf_embeddings = np.load('.\\Dataset\\FlickrDataset\\TFIDF_embeddings\\normalized_tfidf_embeddings.npy')
        faiss_tfidf_index = faiss_indexer(image_paths,captions)
        faiss_tfidf_index.create_index(normalized_tfidf_embeddings)
        
        data_processor = PreprocessData()
        query = data_processor.preprocess_text(query)

        # Load the vectorizer used for creating the TF-IDF embeddings
        vectorizer = joblib.load('.\\tfidf_vectorizer.pkl')
        tfidf_query_embedding = vectorizer.transform([query]).toarray()

        similarities = []
        indices = []
        # Normalize TF-IDF embeddings
        normalized_tfidf_query_embedding = normalize(tfidf_query_embedding, norm='l2')
        if(self.distance_metrice=='cos_similarity'):
            similarities, indices = faiss_tfidf_index.get_nearest_images_cos_sim(normalized_tfidf_query_embedding)
            faiss_tfidf_index.print_results()
            faiss_tfidf_index.save_to_retrieved_folder(dataset_folder)
        return faiss_tfidf_index, similarities, indices