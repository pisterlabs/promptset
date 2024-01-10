import logging
import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.ctransformers import CTransformers

from src import CFG

logging.basicConfig(level=logging.INFO)


def load_mistral() -> CTransformers:
    """Load mistral model."""
    logging.info("Loading mistral model ...")
    model = CTransformers(
        model=os.path.join(CFG.MODELS_DIR, CFG.MISTRAL.MODEL_PATH),
        model_type=CFG.MISTRAL.MODEL_TYPE,
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
