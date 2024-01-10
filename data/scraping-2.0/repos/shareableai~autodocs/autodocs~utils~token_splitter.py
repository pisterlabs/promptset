import tiktoken
from langchain.text_splitter import TokenTextSplitter


def gpt_token_splitter():
    return TokenTextSplitter(chunk_size=3_000, chunk_overlap=250,
                             disallowed_special=(
                                         tiktoken.get_encoding('cl100k_base').special_tokens_set - {'<|endoftext|>'}))
