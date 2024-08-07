from sentence_transformers import SentenceTransformer, InputExample, losses
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

model_name="Sakil/sentence_similarity_semantic_search"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained (".\\model")