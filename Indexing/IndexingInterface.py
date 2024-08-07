from abc import ABC, abstractmethod

class IndexingInterface(ABC):

    @abstractmethod
    def create_index(self,data):
        pass

    
    @abstractmethod
    def print_results(self):
        pass


    @abstractmethod
    def save_to_retrieved_folder(self):
        pass