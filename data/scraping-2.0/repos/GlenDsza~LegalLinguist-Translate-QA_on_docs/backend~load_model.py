# LOAD MODEL
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from constants import MODELS_PATH, CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS, N_BATCH, N_GPU_LAYERS


def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    try:
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
        }
        kwargs["n_gpu_layers"] = N_GPU_LAYERS  # set this based on your GPU

        return LlamaCpp(**kwargs)
    except Exception as e:
        print(e)
        if "ggml" in model_basename:
            logging.INFO(
                "If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None
