from langchain.text_splitter import CharacterTextSplitter

def get_text_chunks(raw_text):
    """
    Splits raw text into smaller chunks.

    Args:
        raw_text (str): The raw text to be split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
