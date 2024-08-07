from RetrievalSystem.IRetrieval import IRetrieval
from abc import abstractmethod
import faiss


class GenericFaissRetrieval(IRetrieval):

    def __init__(self,distance_metrice, embedder):
        self.distance_metrice = distance_metrice
        self.embedder = embedder

    @abstractmethod
    def search(self, query):
        pass
