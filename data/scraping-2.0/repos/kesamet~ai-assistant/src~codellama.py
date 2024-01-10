import logging
import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.ctransformers import CTransformers

from src import CFG

logging.basicConfig(level=logging.INFO)


def load_codellama() -> CTransformers:
    """Load codellama model."""
    logging.info("Loading codellama model ...")
    model = CTransformers(
        model=os.path.join(CFG.MODELS_DIR, CFG.CODELLAMA.MODEL_PATH),
        model_type=CFG.CODELLAMA.MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
            "repetition_penalty": CFG.REPETITION_PENALTY,
            "context_length": CFG.CONTEXT_LENGTH,
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    logging.info("Model loaded")
    return model


def get_prompt(query: str) -> str:
    """
    Generate a prompt based on Llama-2 prompt template.

    Args:
        query (str): The coding problem.

    Returns:
        str: The prompt.
    """
    template = """[INST] Write code to solve the following coding problem that obeys \
the constraints and passes the example test cases. Please wrap your code answer \
using ```:
{query}
[/INST]"""
    return template.format(query=query)
