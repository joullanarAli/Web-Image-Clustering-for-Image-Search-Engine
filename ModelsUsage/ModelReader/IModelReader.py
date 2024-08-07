from abc import ABC, abstractmethod
import numpy as np

class IModelReader(ABC):


    @abstractmethod
    def readModel(self,model_path):
        pass
