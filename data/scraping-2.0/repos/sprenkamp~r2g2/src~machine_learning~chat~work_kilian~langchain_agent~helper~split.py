from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    # separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

def split_text(text: str):
    """
    Split the text into chunks of 1000 characters.

    Args:
        text (str): Text to be split.

    Returns:
        list: List of text chunks.
    """
    return text_splitter.split_documents(text)