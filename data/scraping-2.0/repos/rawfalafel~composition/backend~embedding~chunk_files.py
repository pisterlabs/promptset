from typing import List, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


def get_splitter_for_extension(extension: str) -> RecursiveCharacterTextSplitter:
    """
    Determines the appropriate RecursiveCharacterTextSplitter based on the file extension.

    Parameters:
        extension (str): The file extension (e.g., 'js', 'py', etc.).


    Returns:
        RecursiveCharacterTextSplitter: An instance of a text splitter appropriate for the given file extension.
    """
    if extension in ["js", "ts", "tsx"]:
        return RecursiveCharacterTextSplitter.from_language(Language.JS)
    elif extension == "py":
        return RecursiveCharacterTextSplitter.from_language(Language.PYTHON)
    elif extension == "md":
        return RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)
    elif extension == "html":
        return RecursiveCharacterTextSplitter.from_language(Language.HTML)
    elif extension == "rs":
        return RecursiveCharacterTextSplitter.from_language(Language.RUST)
    else:
        # Fallback to generic RecursiveCharacterTextSplitter if language is not recognized
        return RecursiveCharacterTextSplitter()


def process_chunkfiles(
    chunkfile_data: List[Tuple[str, Union[str, Exception]]]
) -> List[Tuple[str, int, str]]:
    """
    Processes a list of tuples containing file paths and content, and chunks the files using Langchain.

    Parameters:
        chunkfile_data (List[Tuple[str, Union[str, Exception]]]): The return value from generate_chunkfiles().

    Returns:
        List[Tuple[str, int, str]]: A list of tuples, where each tuple has (file_path, chunk_index, chunk_content).
    """
    processed_chunks: List[Tuple[str, int, str]] = []

    for file_path, file_content in chunkfile_data:
        # Check if the content is an Exception and skip processing if so
        if isinstance(file_content, Exception):
            processed_chunks.append((file_path, 0, str(file_content)))
            continue

        # Get the file extension and appropriate text splitter
        ext = file_path.split(".")[-1]
        splitter = get_splitter_for_extension(ext)

        # Split the file content into chunks
        chunks = splitter.create_documents([file_content])

        # Save the chunks in the output list
        for index, chunk in enumerate(chunks):
            processed_chunks.append((file_path, index, chunk.page_content))

    return processed_chunks
