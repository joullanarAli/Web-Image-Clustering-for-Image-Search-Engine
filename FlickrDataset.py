from DatasetReader.IDataset_reader import IDataset_reader
import numpy as np
import pandas as pd

class FlickrDataset_reader(IDataset_reader):


    def __init__(self):
        self.data = '.\\Dataset\\FlickrDataset\\captions.csv'
        
    def readDataset(self):
        self.df = pd.read_csv(self.data)
        self.image_paths = self.df['image'].tolist()
        self.captions = self.df['caption'].tolist()
        return self.df, self.image_paths, self.captions