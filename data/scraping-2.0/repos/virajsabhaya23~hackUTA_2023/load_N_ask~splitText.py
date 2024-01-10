from langchain.text_splitter import CharacterTextSplitter

def split_text(text):
    """
        Splits a text into smaller chunks
        :param text: text to be split
        :return: list of chunks
    """
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks