import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

processor.save_pretrained(".\\PretrainedModels\\ImageCaptioning_models\\CLIP_model")
model.save_pretrained (".\\PretrainedModels\\ImageCaptioning_models\\CLIP_model")