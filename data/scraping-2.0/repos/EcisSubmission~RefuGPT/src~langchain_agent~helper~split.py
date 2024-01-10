from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n \n","\n", ".", "?", "!"],
    chunk_size=512, # 256 for Telegram, 512 for Official Website #change
    chunk_overlap=126,
    length_function=len
)

def split_documents(text: str):
    """
    Split the text into chunks.

    Args:
        text (str): Text to be split.

    Returns:
        list: List of text chunks.
    """
    return text_splitter.split_documents(text)

def create_documents_from_texts(texts: list, metadata: list = None):
    """
    Create a list of documents from a list of texts.

    Args:
        texts (list): List of texts.

    Returns:
        list: List of documents.
    """
    return text_splitter.create_documents(texts, metadatas=metadata)