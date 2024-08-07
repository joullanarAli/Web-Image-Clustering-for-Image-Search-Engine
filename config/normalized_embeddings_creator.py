import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
if '__file__' in globals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'DataPreprocessing')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'Dataset')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'PretrainedModels')))

import numpy as np
import pandas as pd
from datasets import Dataset
import faiss
from transformers import AutoTokenizer, AutoModel
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


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

# Specify the path to your saved model directory
model_path = "PretrainedModels\\model"

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model from the local directory
model = AutoModel.from_pretrained(model_path, from_tf=False, use_safetensors=True)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_batch_embeddings(text_list, batch_size=32):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        batch_embeddings = cls_pooling(model_output).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

#Normalize embeddings
embeddings = get_batch_embeddings(preprocessed_captions)
normalized_embeddings = normalize_embeddings(embeddings)
# Save normalized_embeddings
np.save('.\\preprocessed_normalized_embeddings.npy', normalized_embeddings)
print("Normalized embeddings was saved successfully!")

