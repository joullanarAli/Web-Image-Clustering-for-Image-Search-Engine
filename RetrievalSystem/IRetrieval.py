from abc import ABC, abstractmethod

class IRetrieval(ABC):

    @abstractmethod
    def search(self,query):
        pass
