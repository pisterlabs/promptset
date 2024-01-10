# TextChunkSplitterService will be used to split the text into 
# chunks of different length as specified by the user.
from langchain.text_splitter import CharacterTextSplitter
class TextChunkSplitterService:
    def __init__(self, separator, chunk_size, chunk_overlap, length_function):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text):
        # Split the text into chunks of specified length
        # and return the list of chunks
        text_splitter = CharacterTextSplitter(
        separator=self.separator,
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks