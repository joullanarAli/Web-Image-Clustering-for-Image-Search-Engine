import numpy as np
import pandas as pd
import faiss
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

# Specify the path to the saved model directory
model_path = ".\\model"

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model from the local directory
model = AutoModel.from_pretrained(model_path,from_tf=False, use_safetensors=True)


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings



def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    #encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def get_batch_embeddings(text_list, batch_size=32):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        batch_embeddings = cls_pooling(model_output).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


df = pd.read_csv('.\\captions.csv')

image_paths = df['image'].tolist()
captions = df['caption'].tolist()


# np.array(embeddings).shape
embeddings_dataset = Dataset.load_from_disk('embeddings_dataset1')
embeddings=embeddings_dataset['embeddings']
normalized_embeddings = normalize_embeddings(np.array(embeddings))


dimension = normalized_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity for normalized vectors)
index.add(normalized_embeddings)      # Add normalized embeddings to the index
faiss.write_index(index, 'index_cos_sim.faiss')

# Get embeddings in batches
batch_size = 32
embeddings = get_batch_embeddings(captions, batch_size)
normalized_embeddings = normalize_embeddings(embeddings)
embeddings = np.array(get_embeddings(captions))
normalized_embeddings = normalize_embeddings(embeddings)

# Add normalized embeddings to the index
index.add(normalized_embeddings)