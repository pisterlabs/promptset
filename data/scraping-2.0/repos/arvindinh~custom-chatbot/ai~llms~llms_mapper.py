import os
from langchain.llms import OpenAI

class LLMs_Mapper:
    """
    A class to initalize a new language model based on the wrappers from langchain.
    """
    def __init__(self):
        """
        initializes a new mapper to return a LLM object based on the langchain wrapper

        """
        self.openai_key = os.environ.get('OPENAI_API_KEY')

        self.model_map = {
            "openai" : (OpenAI, {"temperature": 0.7, "openai_api_key": self.openai_key}),
            #temperature:takes values 0-2, lower = more focused and deterministic, higher = random and diverse. 
        }
    
    def find_model(self, model):
        if model in self.model_map:
            model_class, model_args = self.model_map[model]
            model = model_class(**model_args)
            return model
        
        raise ValueError(f"LLM '{model}' not recognized")
