import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pandas as pd
from datasets import Dataset
import faiss

from DataPreprocessing.Preprocess import PreprocessData
from transformers import AutoTokenizer, AutoModel

# Specify the path to the saved model directory
model_path = ".\\model"

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model from the local directory
model = AutoModel.from_pretrained(model_path,from_tf=False, use_safetensors=True)

# Load CSV file
df = pd.read_csv('.\\captions.csv')

image_paths = df['image'].tolist()
captions = df['caption'].tolist()

data_processor = PreprocessData()

preprocessed_captions = []
for caption in captions:
    preprocessed_captions.append(data_processor.preprocess_text(caption))

df['preprocessed_caption'] = preprocessed_captions

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


captions_df = df.explode("preprocessed_caption", ignore_index=True)
caption_dataset = Dataset.from_pandas(captions_df)


embeddings_dataset = caption_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["preprocessed_caption"]).detach().cpu().numpy()[0]}
)

# Save the dataset to disk
output_dir = ".\\embeddings_dataset1"
embeddings_dataset.save_to_disk(output_dir)