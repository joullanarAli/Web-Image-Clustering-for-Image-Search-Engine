#from Indexing.IndexingInterface import IndexingInterface
from Indexing.Generic_indexer import GenericIndexer

import faiss
#import pandas as pd
class faiss_indexer(GenericIndexer):

    def __init__(self,image_paths,captions):
        self.image_paths = image_paths
        self.captions = captions


    def create_index(self,normalized_embeddings):
        self.index = faiss.IndexFlatIP(normalized_embeddings.shape[1])  # Use IndexFlatIP for cosine similarity with normalized embeddings
        self.index.add(normalized_embeddings)
        return self.index
    