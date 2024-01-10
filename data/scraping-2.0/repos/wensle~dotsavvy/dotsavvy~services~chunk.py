from typing import AbstractSet, Collection, Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotsavvy.utils.env_variables import get_env_variable

# Constants
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 20
_ALLOWED_SPECIAL = "all"
_DISALLOWED_SPECIAL = "all"


def get_text_chunks(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    model_name: str | None = None,
    encoding_name: str | None = None,
) -> list[str]:
    """
    Splits the provided text into smaller chunks using the
    RecursiveCharacterTextSplitter.

    The RecursiveCharacterTextSplitter divides text into chunks based on a list of
    characters, prioritizing splitting at paragraph breaks, then at line breaks, spaces,
    and if necessary, any character. This maintains continuity and semantic relationship
    as much as possible.


    """
    # Use the provided values or the default ones
    chunk_size = chunk_size or _CHUNK_SIZE
    chunk_overlap = chunk_overlap or _CHUNK_OVERLAP
    model_name = model_name or get_env_variable("DOTSAVVY_LLM_NAME")
    encoding_name = encoding_name or get_env_variable(
        "DOTSAVVY_TOKENIZER_ENCODING_NAME"
    )

    text_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            model_name=model_name,
        )
    )
    chunks: list[str] = text_splitter.split_text(text)
    return chunks


def create_recursive_tiktoken_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    encoding_name: str | None = None,
    model_name: str | None = None,
    allowed_special: AbstractSet[str] | Literal["all"] | None = None,
    disallowed_special: Collection[str] | Literal["all"] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Returns a LangChain RecursiveCharacterTextSplitter object.

    The RecursiveCharacterTextSplitter divides text into chunks based on a list of
    characters, prioritizing splitting at paragraph breaks, then at line breaks, spaces,
    and if necessary, any character. This maintains continuity and semantic relationship
    as much as possible.

    """
    # Use the provided values or the default ones
    chunk_size = chunk_size or _CHUNK_SIZE
    chunk_overlap = chunk_overlap or _CHUNK_OVERLAP
    encoding_name = encoding_name or get_env_variable(
        "DOTSAVVY_TOKENIZER_ENCODING_NAME"
    )
    model_name = model_name or get_env_variable("DOTSAVVY_LLM_NAME")
    allowed_special = allowed_special or _ALLOWED_SPECIAL
    disallowed_special = disallowed_special or _DISALLOWED_SPECIAL

    text_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
    )
    return text_splitter
