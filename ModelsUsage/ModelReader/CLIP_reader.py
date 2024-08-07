from transformers import CLIPProcessor, CLIPModel
from ModelsUsage.ModelReader.IModelReader import IModelReader

class CLIP_reader(IModelReader):


    def __init__(self):
        self.model_path = ".\\.\\PretrainedModels\\ImageCaptioning_models\\CLIP_model"
        

    def readModel(self):
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model = CLIPModel.from_pretrained(self.model_path)
        return self.processor, self.model