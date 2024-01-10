#CreateEmbeddingService


# This service will be used to create embeddings for the given data.

#CreateEmbeddingService  shall create an object either of OpenEMbeddings or any other bert uncased embedding model
# and then call the create_embeddings method of that object to create embeddings for the given data.
from langchain.embeddings.openai import OpenAIEmbeddings
from services.GetEnvironmentVariables import GetEnvVariables
import os
#import transformers

# get env variables 
env_vars = GetEnvVariables()
OPENAI_API_KEY = env_vars.get_env_variable('openai_key')
# insert your API_TOKEN here
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class CreateEmbeddingService:
    def __init__(self):
        pass

    def create_embeddings(self, embedding_model):
        # Logic to create embeddings for the given data
        # Replace this with the actual implementation
        embeddings = EmbeddingModelFactory().get_embedding_model(embedding_model)
       
        return embeddings
    
class EmbeddingModelFactory:
    def __init__(self):
        pass
    
    def get_embedding_model(self, embedding_model_name):
        # Logic to create the embedding model object
        if embedding_model_name == "OpenEmbeddings":
            embedding_model = OpenAIEmbeddings(model="gpt-3.5-turbo")
            return embedding_model
        elif embedding_model_name == "BertUncased":
            embedding_model = transformers.AutoModel.from_pretrained("bert-base-uncased")
            return embedding_model
        else:
            raise ValueError("Unsupported embedding model")