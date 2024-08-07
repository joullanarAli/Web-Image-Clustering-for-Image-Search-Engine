from transformers import BlipProcessor, BlipForConditionalGeneration
from ModelsUsage.ModelReader.IModelReader import IModelReader

class BLIP_reader(IModelReader):


    def __init__(self):
        self.model_path = ".\\.\\PretrainedModels\\ImageCaptioning_models\\BLIP_model"
        

    def readModel(self):
        self.processor = BlipProcessor.from_pretrained(self.model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_path)
        return self.processor, self.model