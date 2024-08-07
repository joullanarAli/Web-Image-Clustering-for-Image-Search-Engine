import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
class IDataset_reader(ABC):


    @abstractmethod
    def readDataset(self):
        pass