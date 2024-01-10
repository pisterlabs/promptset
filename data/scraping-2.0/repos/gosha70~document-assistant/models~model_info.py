
from langchain.embeddings import HuggingFaceInstructEmbeddings
from .models_constants import (
    DEFAULT_MODEL_BASENAME, 
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME, 
    DEVICE_TYPE_CPU,
    EMBEDDING_KWARGS,
    ENCODE_KWARG
)

"""
ModelInfo class defines the setting for creating and training LLM used in the retrieval framework.
    - model_name (str) 
    - model_id (str) 
    - model_basename (str) 
    - device_type (str): 'cpu', 'cuda', or 'mps'

See: https://huggingface.co/docs/transformers/model_doc/auto    
"""
class ModelInfo:
    def __init__(self):
        self._model_name = DEFAULT_MODEL_NAME
        self._model_id = DEFAULT_MODEL_ID
        self._model_basename = DEFAULT_MODEL_BASENAME
        self._device_type = DEVICE_TYPE_CPU # the type of device where the model runs: 'cpu', 'cuda'

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    @property
    def model_basename(self):
        return self._model_basename

    @model_basename.setter
    def model_basename(self, value):
        self._model_basename = value

    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, value):
        self._device_type = value

    def __str__(self):
        return (f"ModelInfo(model_name='{self._model_name}', "
                f"model_id='{self._model_id}', "
                f"model_basename='{self._model_basename}', "
                f"device_type='{self._device_type}')")    
    
    def create_embedding(model_name):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        return HuggingFaceInstructEmbeddings(
            model_name=model_name, 
            model_kwargs=EMBEDDING_KWARGS, 
            encode_kwargs=ENCODE_KWARG
        )
    
    def embedding_class():
        return "langchain.embeddings.HuggingFaceInstructEmbeddings"   