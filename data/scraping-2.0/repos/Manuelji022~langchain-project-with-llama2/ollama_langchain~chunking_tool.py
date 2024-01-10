from langchain.text_splitter import CharacterTextSplitter

class chunking_tool():
    """
    Class for splitting text into chunks.
    Parameters:
        text (str): The text to split into chunks.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
    """
    def __init__(self, text, chunk_size, chunk_overlap):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function = len
            )
        
        chunks = text_splitter.split_text(self.text)
        return chunks
