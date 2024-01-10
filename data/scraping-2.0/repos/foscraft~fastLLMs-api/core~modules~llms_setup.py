from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import (
    GPT4All,
    LlamaCpp,
)

from .config import get_env_variable


def setup_gpt4all(llm_params):
    """to handle the setup of the Language Model Chains (GPT4All and LlamaCpp).
    This will help keep the LLM setup code organized.
    """
    model_path = get_env_variable("MODEL_PATH")
    model_n_ctx = get_env_variable("MODEL_N_CTX")
    model_n_batch = int(get_env_variable("MODEL_N_BATCH", 8))  # type: ignore
    callbacks = [StreamingStdOutCallbackHandler()]
    return GPT4All(
        model=model_path,
        # n_ctx=model_n_ctx,
        backend="gptj",
        n_batch=model_n_batch,
        callbacks=callbacks,
        verbose=False,
        **llm_params
    )


def setup_llama_cpp(llm_params):
    """to handle the setup of the Language Model Chains (GPT4All and LlamaCpp).
    This will help keep the LLM setup code organized.
    """
    model_path = get_env_variable("MODEL_PATH")
    model_n_ctx = get_env_variable("MODEL_N_CTX")
    model_n_batch = int(get_env_variable("MODEL_N_BATCH", 8))  # type: ignore
    callbacks = [StreamingStdOutCallbackHandler()]  # type: ignore
    return LlamaCpp(
        model_path=model_path,
        # n_ctx=model_n_ctx,
        n_batch=model_n_batch,
        callbacks=callbacks,
        verbose=False,
        **llm_params
    )
