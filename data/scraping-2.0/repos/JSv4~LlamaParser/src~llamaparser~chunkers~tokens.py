import enum
import typing

from langchain.text_splitter import TokenTextSplitter


class Tokenizers(str, enum.Enum):
    OPENAI = "OPENAI"


def split_text_into_chunks_by_tokens(
        text: str,
        tokenizer: Tokenizers = Tokenizers.OPENAI,
        tokens_per_chunk: int = 500,
        token_overlap: int = 50
) -> typing.List[str]:

    if tokenizer == Tokenizers.OPENAI:
        text_splitter = TokenTextSplitter(chunk_size=tokens_per_chunk, chunk_overlap=token_overlap)
        texts = text_splitter.split_text(text)
    else:
        raise ValueError("Only OpenAI tokenizer currently supported...")

    return texts
