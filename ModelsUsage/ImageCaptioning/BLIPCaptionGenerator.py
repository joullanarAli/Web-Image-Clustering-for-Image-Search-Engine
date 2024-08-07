from ModelsUsage.ImageCaptioning.ICaptionGenerator import ICaptionGenerator
from ModelsUsage.ModelReader.BLIP_reader import BLIP_reader
from PIL import Image

class BLIPCaptionGenerator(ICaptionGenerator):

    def __init__(self,dataset):
        BLIP_model = BLIP_reader()
        self.processor, self.model = BLIP_model.readModel()
        if dataset == "flickr" or dataset == "Flickr":
            self.dataset_path = ".\\Dataset\\FlickrDataset\\Images"

    def generateCaption(self,image):
        path = self.dataset_path+"\\"+image
        image = Image.open(path)
        inputs = self.processor(images=image, return_tensors="pt")
         # Set generation parameters for longer and more detailed captions
        generation_args = {
            "max_length": 100,  # Increase the maximum length of the caption
            "num_beams": 5,     # Use beam search with a specified number of beams
            "temperature": 1.0, # Control randomness (1.0 is the default value)
            "top_k": 50,        # Consider the top k tokens
            "top_p": 0.95,      # Nucleus sampling
            "no_repeat_ngram_size": 2  # Avoid repeating the same n-grams
        }

        out = self.model.generate(**inputs, **generation_args)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption