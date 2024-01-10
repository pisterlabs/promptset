import tiktoken
import base64
from config.openai import openai_config

# for a given content string, count the number of tokens if it were to be embedded


async def count_embedding_tokens(payload: str):
    encoding = tiktoken.get_encoding(
        openai_config.embedding_token_encoder_name)
    return len(encoding.encode(payload))

# for a given content string, count the number of tokens if it were to be used in the chat completion model
# TODO: improve here looking at https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


async def count_embedding_tokens(payload: str):
    encoding = tiktoken.get_encoding(
        openai_config.completion_token_encoder_name)
    return len(encoding.encode(payload))

# determines whether the request would be processed furthur or not
# this includes many check conditions including the max permitted token count


async def determine_request_validity():
    return True


async def base64_encode_string(payload: str):
    return base64.b64encode(payload.encode()).decode()


async def base64_decode_string(payload: str):
    return base64.b64decode(payload).decode()
