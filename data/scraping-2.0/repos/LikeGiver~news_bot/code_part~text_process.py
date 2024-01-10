import os
from langchain.text_splitter import CharacterTextSplitter

def split_text_file(text_string, max_chars_per_file=400, chunk_overlap=0, separator="\n\n"):

    text_splitter = CharacterTextSplitter(
        separator = separator,
        chunk_size = max_chars_per_file,
        chunk_overlap  = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
    )

    text_chunks = text_splitter.create_documents([text_string])

    return text_chunks
