import logging
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from .models_constants import (
    N_CTX, 
    N_BATCH, 
    N_GPU_LAYERS, 
    DEVICE_TYPE_MPS, 
    DEVICE_TYPE_CUDA
)

"""
Creates a GGUF(previously GGML) quantization method that allows users to use the CPU to run 
an LLM but also offload some of its layers to the GPU for a speed up.

This function relies on the LlamaCpp library to load a GGUF quantized model.

Parameters:
- model_info (ModelInfo): the class storing the information about LLM:
     model_name (str) 
     model_id (str) 
     model_basename (str) 
     device_type (str)
- cache_dir (str): The path to the local cache directory where loaded models are stored.

Returns:
- LlamaCpp: The LlamaCpp model if successful, otherwise - None.
"""
def load_gguf_model(model_info, cache_dir):    

    try:
        model_path = hf_hub_download(
            repo_id=model_info.model_id,
            filename=model_info.model_basename,
            resume_download=True,
            cache_dir=cache_dir,
        )
        params = {
            "model_path": model_path,
            "n_ctx": N_CTX,
            "max_tokens": N_CTX,
            "n_batch": N_BATCH,  
            "verbose": True,
        }
        if model_info.device_type.lower() == DEVICE_TYPE_MPS:
            params["n_gpu_layers"] = 1
        if model_info.device_type.lower() == DEVICE_TYPE_CUDA:
            params["n_gpu_layers"] = N_GPU_LAYERS

        logging.info(f"Creating (LlamaCpp) with the arguments: '{params}' ...")

        return LlamaCpp(**params)
    except Exception as error:
        logging.error(f"Failed to create (LlamaCpp) for '{model_info}': {str(error)}", exc_info=True)
        return None