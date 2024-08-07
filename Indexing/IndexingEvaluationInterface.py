from abc import ABC, abstractmethod


class IndexingEvaluationInterface(ABC):

    @abstractmethod
    def evaluate_index(self,index):
        pass