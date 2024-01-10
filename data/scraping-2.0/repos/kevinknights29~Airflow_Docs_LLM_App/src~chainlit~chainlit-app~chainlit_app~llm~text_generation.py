from __future__ import annotations

import os

from chainlit_app.common import config
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


LLM_CHAIN = None


def _build_prompt(template=None, input_variables=None):
    if template is None and input_variables is None:
        template = (
            "Question: {question}\n"
            "\n"
            "Answer: Let's work this out in a step by step way to be sure we have the right answer."
        )
        input_variables = ["question"]
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    return prompt


def _llm_init(model_params=None):
    if model_params is None:
        model_params = {
            "temperature": 0.1,
            "max_length": 2000,
            "top_p": 1,
        }

    _model_path = config()["model"]
    _n_gpu_layers = os.environ.get(
        "N_GPU_LAYERS",
        40,
    )  # Change this value based on your model and your GPU VRAM pool.
    _n_batch = os.environ.get(
        "N_BATCH",
        512,
    )  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    # Callbacks support token-wise streaming
    _callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=_model_path,
        input=model_params,
        n_gpu_layers=_n_gpu_layers,
        n_batch=_n_batch,
        callback_manager=_callback_manager,
        verbose=True,
    )
    return llm


def llm_chain():
    global LLM_CHAIN
    if LLM_CHAIN is None:
        _llm = _llm_init()
        llm_chain = LLMChain(prompt=_build_prompt(), llm=_llm)
        LLM_CHAIN = llm_chain
    return LLM_CHAIN
