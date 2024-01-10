# import OpenAI token encoding module
from tiktoken import Encoding

# function which counts tokens
async def count_tokens(encoding: Encoding, text: str) -> int:

    # returns an int
    return len(encoding.encode(text))