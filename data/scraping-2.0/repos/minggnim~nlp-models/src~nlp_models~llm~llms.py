from langchain.llms import CTransformers
from .base import LlmConfig


def build_llm(cfg: LlmConfig):
    llm = CTransformers(
        model=cfg.MODEL_BIN_PATH,
        model_type=cfg.MODEL_TYPE,
        config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                'temperature': cfg.TEMPERATURE}
    )
    return llm
