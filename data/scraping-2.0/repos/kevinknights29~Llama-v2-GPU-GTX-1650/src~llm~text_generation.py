from __future__ import annotations

import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from src.utils import utils


def _llm_init(**kwargs):
    model_params = {}
    model_params["temperature"] = kwargs.get("temperature", 0.1)
    model_params["max_length"] = kwargs.get("max_length", 2000)
    model_params["top_p"] = kwargs.get("top_p", 0.95)
    model_params["top_k"] = kwargs.get("top_p", 40)
    
    _model_path = utils.find_model()
    _n_gpu_layers = os.environ.get(
        "N_GPU_LAYERS",
        35,
    )  # Change this value based on your model and your GPU VRAM pool.
    _n_batch = os.environ.get(
        "N_BATCH",
        512,
    )  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    _n_threads = os.environ.get(
        "N_THREADS",
        4,
    )  # Change this value based on your CPU cores.
    # Callbacks support token-wise streaming
    _callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=_model_path,
        callback_manager=_callback_manager,
        verbose=True,
        # model parameters
        n_gpu_layers=_n_gpu_layers,
        n_batch=_n_batch,
        n_threads=_n_threads,
        use_mmap=True,
        use_mlock=True,
        temperature=model_params["temperature"],
    )
    return llm


def generate_text(prompt, **kwargs):
    response = _llm_init(**kwargs).stream(prompt)
    return response
