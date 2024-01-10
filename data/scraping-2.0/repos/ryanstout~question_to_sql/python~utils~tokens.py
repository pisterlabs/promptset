import tiktoken

from python.utils.openai import openai_engine

engine = openai_engine()

# Only load the encoder once
token_encoder = tiktoken.encoding_for_model(engine)


def count_tokens(text: str) -> int:
    return len(token_encoder.encode(text))
