from Indexing.IndexingInterface import IndexingInterface
from Indexing.IndexingEvaluationInterface import IndexingEvaluationInterface
from abc import ABC , abstractmethod
from DatasetReader.FlickrDataset import FlickrDataset_reader
from ModelsUsage.Embeddings import Embeddings
from ModelsUsage.ModelReader.sen_sim_sem_search_reader import sen_sim_sem_search_reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from DataPreprocessing.Preprocess import PreprocessData
import pandas as pd
import pandas as pd
from PIL import Image
import os
import shutil
import joblib

class GenericIndexer(IndexingInterface,IndexingEvaluationInterface,ABC):
    

    def __init__():
        pass
    
    @abstractmethod
    def create_index(self,data):
        
        pass

    def get_nearest_images_cos_sim(self, normalized_query_embedding):
        k=2
        self.similarities = []
        self.indices= []
        self.difference = 1
        while self.difference > 0.6 and k<500:
            del self.similarities, self.indices
            k= k+1
            self.similarities, self.indices = self.index.search(normalized_query_embedding, k) 
            self.difference = self.similarities[0][k-1]/self.similarities[0][0] 
        return self.similarities, self.indices
    
    def evaluate_nearest_images_cos_sim(self, normalized_query_embedding,k=100):
        self.similarities, self.indices = self.index.search(normalized_query_embedding, k) 
        return self.similarities, self.indices
    

    def print_results(self):
        samples = {
            "caption": [self.captions[i] for i in self.indices[0]],
            "image": [self.image_paths[i] for i in self.indices[0]],
            "similarities": self.similarities[0],
        }
        self.samples_df = pd.DataFrame.from_dict(samples)
        self.samples_df["similarities"] = self.similarities[0]
        self.samples_df.sort_values("similarities", ascending=False, inplace=True)
        for _, row in self.samples_df.iterrows():
            print(f"Caption: {row.caption}")
            print(f"Similarity: {row.similarities}")
            print(f"Image: {row.image}")
            print("=" * 50)
            print()

    def getSamplesDF(self):
        return self.samples_df

    def save_to_retrieved_folder(self,dataset):
        
        if dataset == 'flickr':
            dataset_folder = 'FlickrDataset'
        images_folder = '.\\Dataset\\'+dataset_folder+'\\Images'
        retrieved_folder = '.\\static\\retrieved_faiss'

        # Ensure the retrieved_faiss directory exists
        os.makedirs(retrieved_folder, exist_ok=True)

        for _, row in self.samples_df.iterrows():

            # Construct the full path to the image file
            image_path = os.path.join(images_folder, row.image)

            # Open and save the image to the retrieved_faiss folder
            if os.path.exists(image_path):
                image = Image.open(image_path)
                #image.show()
                # Construct the path to save the image in the retrieved_faiss folder
                save_path = os.path.join(retrieved_folder, row.image)
                image.save(save_path)
                print(f"Image saved to {save_path}")
            else:
                print(f"Image file {image_path} not found.")
            print()  # Add a blank line for better readability


    def evaluate_index(self,k):
        sen_model = sen_sim_sem_search_reader()
        sen_model.readModel()
        
        dataset = FlickrDataset_reader()   
        df, image_paths, captions, BLIP_captions = dataset.read_BLIPDetailedDataset()
        
        unique_df = df.drop_duplicates(subset=['image'])
        BLIP_captions = unique_df['blip_caption']
        unique_images = unique_df['image']

        total_queries = len(BLIP_captions)
        correct_retrievals = 0

        for i, blip_caption in enumerate(BLIP_captions):
            query = blip_caption
            true_image = unique_images.iloc[i]
            data_processor = PreprocessData()
            query = data_processor.preprocess_text(query)
            query_embedding = sen_model.get_batch_embeddings([query], batch_size=1)
            normalized_query_embedding = sen_model.normalize_embeddings_fun(query_embedding)
            
            similarities, results = self.evaluate_nearest_images_cos_sim(normalized_query_embedding,k)

            samples = {
                "caption": [self.captions[i] for i in results[0]],
                "image": [self.image_paths[i] for i in results[0]],
                "similarities": similarities[0],
            }
            
            samples_df = pd.DataFrame.from_dict(samples)
            samples_df["similarities"] = similarities[0]
            samples_df.sort_values("similarities", ascending=False, inplace=True)
            
            retrieved_images = samples_df["image"].tolist()
            
            if true_image in retrieved_images:
                correct_retrievals += 1

            

        precision = correct_retrievals / total_queries
        print(f"Precision: {precision * 100:.2f}%")

        return precision
    

    def evaluate_tfidf_index(self,k):
        
        
        dataset = FlickrDataset_reader()   
        df, image_paths, captions, BLIP_captions = dataset.read_BLIPDetailedDataset()
        
        unique_df = df.drop_duplicates(subset=['image'])
        BLIP_captions = unique_df['blip_caption']
        unique_images = unique_df['image']

        total_queries = len(BLIP_captions)
        correct_retrievals = 0

        for i, blip_caption in enumerate(BLIP_captions):
            query = blip_caption
            true_image = unique_images.iloc[i]

            # Create TF-IDF vectorizer
            vectorizer = joblib.load('.\\tfidf_vectorizer.pkl')
            tfidf_query_embeddings = vectorizer.transform([query]).toarray()

            # Normalize TF-IDF embeddings
            normalized_tfidf_embeddings = normalize(tfidf_query_embeddings, norm='l2')
            
            similarities, results = self.evaluate_nearest_images_cos_sim(normalized_tfidf_embeddings,k)
            
            # Filter results to get only the top 3 images
            

            samples = {
                "caption": [self.captions[i] for i in results[0]],
                "image": [self.image_paths[i] for i in results[0]],
                "similarities": similarities[0],
            }
            
            samples_df = pd.DataFrame.from_dict(samples)
            samples_df["similarities"] = similarities[0]
            samples_df.sort_values("similarities", ascending=False, inplace=True)
            
            retrieved_images = samples_df["image"].tolist()
            
            if true_image in retrieved_images:
                correct_retrievals += 1

            # For debugging and verification, print the results
            # print(f"Query: {query}")
            # for _, row in samples_df.iterrows():
            #     print(f"Caption: {row.caption}")
            #     print(f"Similarity: {row.similarities}")
            #     print(f"Image: {row.image}")
            #     print("=" * 50)
            #     print()

        precision = correct_retrievals / total_queries
        print(f"Precision: {precision * 100:.2f}%")

        return precision