import asyncio
import hashlib
from typing import AsyncIterable, BinaryIO, Union
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import io
import replicate
from fastapi import HTTPException
from chromadb.api.types import Documents

from .schemas import Message

SYSTEM_PROMPT = """System: Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer."""


async def create_input(messages: list[Message]) -> str:
    """Create the input string for the model.

    :param messages: A list of messages from the user and the system.
    :return: The input string for the model.
    """
    input_string = SYSTEM_PROMPT
    for message in messages:
        input_string += f"\n\n{message.role}: {message.content}"
    return input_string


async def stream_chat(user_input: str) -> AsyncIterable[str | bytes]:
    try:
        # The replicate/llama-2-70b-chat model can stream output as it's running.
        # The predict method returns an iterator, and you can iterate over that output.
        output = replicate.run(
            "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
            input={
                "prompt": f"{SYSTEM_PROMPT}\n\nHuman: {user_input}\nSystem:",
            },
        )
        for item in output:
            # print(item)
            # Dumb as hell - streaming only works if you introduce a tiny delay
            # Hypothesis
            # The delay essentially serves as a workaround by allowing the client to catch up and request the
            # next chunk of data. It's worth investigating further if there are other underlying issues,
            # such as buffering in your server or any intermediaries, that might affect real-time streaming.
            await asyncio.sleep(0.000001)  # Introducing a delay
            yield item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_pdf_text(pdf_doc: bytes) -> str:
    """
    Extracts text from a given PDF document represented as bytes.

    Args:
        pdf_doc (bytes): The PDF document in bytes format.

    Returns:
        str: The extracted text from the PDF document.

    Raises:
        PdfReadError: If there's an error reading the PDF.

    Examples:
        >>> pdf_path = "path_to_pdf_document.pdf"
        >>> with open(pdf_path, 'rb') as f:
        >>>     pdf_bytes = f.read()
        >>> text = get_pdf_text(pdf_bytes)
        >>> print(text[:100])  # Print the first 100 characters of extracted text.
    """
    text = ""
    pdf_reader = PdfReader(io.BytesIO(pdf_doc))
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text: str) -> Documents:
    """Splits a given text into chunks of specified size with a specified
    overlap.

    Args:
        text (str): The text to be split into chunks.

    Returns:
        list: A list of text chunks.

    Example:
        >>> text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, urna id aliquet lacinia, nunc nisl ultrices nunc, id lacinia nunc nisl id nisl."
        >>> get_text_chunks(text)
        ['Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, urna id aliquet lacinia, nunc nisl ultrices nunc,', 'id lacinia nunc nisl id nisl.']
    """
    # This doesn't actually work corretly, only splits on newlines
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    return chunks


def compute_md5(data: Union[bytes, BinaryIO]) -> str:
    """
    Compute the MD5 hash of the contents of a bytes object or file-like object.

    Args:
        data (Union[bytes, BinaryIO]): A bytes object or file-like object that supports
                                       binary read and seek operations.

    Returns:
        str: The MD5 hash of the contents.
    """

    md5 = hashlib.md5()

    if isinstance(data, bytes):
        md5.update(data)
    else:
        # Read the file in chunks to avoid using too much memory
        for chunk in iter(lambda: data.read(4096), b""):
            md5.update(chunk)

        # Reset the file pointer to its beginning
        data.seek(0)

    return md5.hexdigest()
