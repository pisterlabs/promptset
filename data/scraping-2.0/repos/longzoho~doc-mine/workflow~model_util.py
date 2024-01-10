import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain import LlamaCpp, OpenAI

from util.file_util import get_file_path_from_root

load_dotenv()

model_path = hf_hub_download(repo_id='TheBloke/Llama-2-7b-Chat-GGUF', filename='llama-2-7b-chat.Q4_K_M.gguf')
# model_path = get_file_path_from_root("models/llama-2-7b-chat.gguf")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
if device_type.lower() == 'cpu':
    print('Warning: CUDA is not available, using CPU instead.')


def create_llama_model():
    kwargs = {
        "model_path": model_path,
        "n_ctx": 2048 * 4,
        "max_tokens": 2048 * 4,
        "n_gpu_layers": 1000 if device_type.lower() in ["mps", "cuda"] else None,
        "n_batch": 2048 if device_type.lower() == "cuda" else None,
        "verbose": False,
        "f16_kv": True,
    }

    # create lager language model
    return LlamaCpp(**kwargs)


def create_openai_model():
    return OpenAI()
