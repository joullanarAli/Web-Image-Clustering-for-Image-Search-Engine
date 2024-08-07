from abc import ABC, abstractmethod
import numpy as np

class ICaptionGenerator(ABC):

    @abstractmethod
    def generateCaption(self):
        pass