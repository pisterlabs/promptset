import os
from langchain.llms import (OpenAI, LlamaCpp)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (OpenAIEmbeddings, HuggingFaceInstructEmbeddings)

import shared.constants as constants

def get_chat_model(local, ai_temp = constants.AI_TEMP, max_tokens = None): 
    if local:                  
        raise Exception("Chat model not supported locally")
    else:
        openai_api_key = get_openai_api_key()
        return ChatOpenAI(temperature=ai_temp, openai_api_key=openai_api_key)#, max_tokens=max_tokens) ## Don't use max_tokens for now

def get_llm(local, local_model_path = None, ai_temp = constants.AI_TEMP, max_tokens = -1):  
    """Create an LLM

    Args:
        local (bool): Create a local or remote LLM
        local_model_path (str, optional): Path to the local model, if running locally. If None, defaults to constants.LOCAL_MODEL_PATH.
        ai_temp (float, optional): AI Temp. Defaults to constants.AI_TEMP.
        max_tokens (int, optional): Max token count. Defaults to -1.
    """
    if local:          
        
        if os.environ.get("OFFLOAD_TO_GPU_LAYERS") == None: 
            offload_layers = constants.OFFLOAD_TO_GPU_LAYERS
        else:
            offload_layers = os.environ.get("OFFLOAD_TO_GPU_LAYERS")

        if local_model_path == None:
            local_model_path = constants.DEFAULT_LOCAL_MODEL_PATH

        return LlamaCpp(model_path=local_model_path, n_ctx=constants.MAX_LOCAL_CONTEXT_SIZE, max_tokens=max_tokens, temperature=ai_temp, n_gpu_layers=offload_layers) 
    else:
        openai_api_key = get_openai_api_key()
        return OpenAI(temperature=ai_temp, openai_api_key=openai_api_key, max_tokens=max_tokens)
    
def get_embedding(local, device_type = "cpu"):    
    if local:
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device_type}) #, embed_instruction="Represent the document for retrieval: ")
    else:
        openai_api_key = get_openai_api_key()
        return OpenAIEmbeddings(openai_api_key=openai_api_key)  
    
def get_openai_api_key():
    from dotenv import dotenv_values, load_dotenv
    load_dotenv()
    return dotenv_values().get('OPENAI_API_KEY')