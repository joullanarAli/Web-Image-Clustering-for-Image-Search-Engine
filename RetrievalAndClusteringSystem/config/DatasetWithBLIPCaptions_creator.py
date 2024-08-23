import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

if '__file__' in globals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'Dataset')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'DatasetReader')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'ModelsUsage')))

from ModelsUsage.ModelReader.BLIP_reader import BLIP_reader
import pandas as pd
import numpy as np
from DatasetReader.FlickrDataset import FlickrDataset_reader


dataset = FlickrDataset_reader()
df, image_paths, captions= dataset.readDataset()

BLIP_model = BLIP_reader()
processor, model = BLIP_model.readModel()

from ModelsUsage.ImageCaptioning.BLIPCaptionGenerator import BLIPCaptionGenerator
BLIPCaptionGeneratorObj = BLIPCaptionGenerator('flickr')

unique_images = df.drop_duplicates(subset=['image'])

from tqdm import tqdm
tqdm.pandas()   # Initialize tqdm for progress bar
unique_images['blip_caption'] = unique_images['image'].progress_apply(BLIPCaptionGeneratorObj.generateCaption)
unique_images = pd.DataFrame(unique_images)
final_dataset = pd.merge(df, unique_images[['image', 'blip_caption']], on='image', how='left')

final_dataset.to_csv('.\\Dataset\\FlickrDataset\\flickr_dataset_with_BLIP_detailedtemp_captions.csv', index=False)

