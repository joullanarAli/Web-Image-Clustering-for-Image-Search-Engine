#from abc import ABC, abstractmethod
import numpy as np

class Embeddings():

    def __init__():
        pass

    def cls_pooling(self,model_output):
        return model_output.last_hidden_state[:, 0]


    def get_text_embeddings(self,text_list,tokenizer):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        #encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)

    def get_batch_embeddings(self,text_list, batch_size=32):
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            encoded_input = {k: v for k, v in encoded_input.items()}
            model_output = self.model(**encoded_input)
            batch_embeddings = self.cls_pooling(model_output).detach().cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def normalize_embeddings_fun(self,embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings
        