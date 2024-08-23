import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib

# Ensure the current directory and parent directory are included in the Python path
if '__file__' in globals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'DataPreprocessing')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'Dataset')))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from DataPreprocessing.Preprocess import PreprocessData

# Load CSV file
df = pd.read_csv('Dataset\\FlickrDataset\\captions.csv')

image_paths = df['image'].tolist()
captions = df['caption'].tolist()

data_processor = PreprocessData()

preprocessed_captions = []
for caption in captions:
    preprocessed_captions.append(data_processor.preprocess_text(caption))

df['preprocessed_caption'] = preprocessed_captions

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed captions
tfidf_embeddings = vectorizer.fit_transform(df['preprocessed_caption']).toarray()
# Normalize TF-IDF embeddings
normalized_tfidf_embeddings = normalize(tfidf_embeddings, norm='l2')


# Save the vectorizer used for indexing
joblib.dump(vectorizer, '.\\preprocessed_tfidf_vectorizer.pkl')

# Save normalized TF-IDF embeddings
#np.save('.\\Dataset\\FlickrDataset\\TFIDF_embeddings\\normalized_tfidf_embeddings.npy', normalized_tfidf_embeddings)
print("Normalized TF-IDF embeddings were saved successfully!")
