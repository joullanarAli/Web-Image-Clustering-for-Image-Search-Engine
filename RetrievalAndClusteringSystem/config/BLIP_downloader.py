from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

#pip install transformers pillow requests
import requests



# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

processor.save_pretrained(".\\PretrainedModels\\ImageCaptioning_models\\BLIP_model")
model.save_pretrained(".\\PretrainedModels\\ImageCaptioning_models\\BLIP_model")