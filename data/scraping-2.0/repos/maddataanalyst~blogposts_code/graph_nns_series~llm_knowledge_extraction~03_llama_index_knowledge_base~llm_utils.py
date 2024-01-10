import os
import torch
from llama_index.llms import HuggingFaceLLM, AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
    SimpleDirectoryReader,
    KnowledgeGraphIndex
)
from typing import Tuple
from huggingface_hub import login


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
EMBEDDING_MODEL = "local:BAAI/bge-small-en"

def setup_local_llama(
        model_name: str = LLAMA2_7B_CHAT,
        tokenizer_model: str = LLAMA2_7B_CHAT,
        embedding_model: str = EMBEDDING_MODEL,
        context_window: int = 4096,
        max_new_tokens: int = 2048,
        device_map: str = "cuda:0",
        model_kwargs: dict = {"torch_dtype": torch.float16, "load_in_4bit": True},
        **kwargs) -> Tuple[HuggingFaceLLM, object]:
    """
    Set up a local instance of the HuggingFaceLLM class for generating text using the Llama language model.
    All proper agruments should be passed to function.
    
    Args:
        model_name (str): The name of the pre-trained Llama model to use.
        tokenizer_model (str): The name of the pre-trained tokenizer to use.
        context_window (int): The number of tokens to use as context when generating new text.
        max_new_tokens (int): The maximum number of new tokens to generate when generating new text.
        device_map (str): The device to use for running the model (e.g. "cuda:0" for GPU or "cpu" for CPU).
        model_kwargs (dict): Additional keyword arguments to pass to the Llama model.
        **kwargs: Additional keyword arguments to pass to the HuggingFaceLLM constructor.
    
    Returns:
        HuggingFaceLLM: A local instance of the HuggingFaceLLM class for generating text using the Llama language model.
    """
    login()
    llm = HuggingFaceLLM(
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        tokenizer_name=tokenizer_model,
        model_name=model_name,
        device_map=device_map,
        # change these settings below depending on your GPU
        model_kwargs=model_kwargs,
        **kwargs.get('llm_kwargs', {}))
    
    return llm, EMBEDDING_MODEL


def setup_azure_llm(*args, **kwargs) -> AzureOpenAI:
    """
    Sets up an instance of AzureOpenAI for use with the Llama Language Model.
    
    Required environmental variables:
    - ENGINE_NAME: the name of the engine to use (GPT-4, GPT-35, etc.)
    - OPENAI_MODEL: the name of the OpenAI model to use
    - OPENAI_API_KEY: the API key for the OpenAI API
    - OPENAI_API_BASE: the base URL for the OpenAI API
    - OPENAI_API_TYPE: the type of the OpenAI API (e.g. "davinci" or "curie")
    """

    llm = AzureOpenAI(
        engine=os.environ['ENGINE_NAME'],
        model=os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ["OPENAI_API_BASE"],
        api_type=os.environ["OPENAI_API_TYPE"], 
        *args, **kwargs
    )
    return llm

def setup_azure_embedding_model(*args, **kwargs) -> OpenAIEmbedding:
    """
    Sets up and returns an instance of the OpenAIEmbedding class for Azure deployment.
    
    Required environment variables:
    - EMBEDDING_DEPLOYMENT_NAME: The name of the Azure deployment for the embedding model.
    - OPENAI_API_KEY: The API key for the OpenAI API.
    - OPENAI_API_BASE: The base URL for the OpenAI API.
    - OPENAI_API_TYPE: The type of the OpenAI API.
    """
    embed = OpenAIEmbedding(
        deployment_name=os.environ["EMBEDDING_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ["OPENAI_API_BASE"],
        api_type=os.environ["OPENAI_API_TYPE"],
    )
    return embed


def setup_azure_openai(*args, **kwargs) -> Tuple[AzureOpenAI, object]:
    azure_llm = setup_azure_llm(*args, **kwargs)
    azure_embedding = setup_azure_embedding_model(*args, **kwargs)
    return azure_llm, azure_embedding