from langchain.text_splitter import RecursiveCharacterTextSplitter

class RecursiveCharacter_TextSplitter:
    """
    A class to split a Document using the RecursiveCharacterTextSplitter wrapper from the langchain library.
    Recommended text splitter for generic text.
    """
    def __init__(self, chunk_size, chunk_overlap, length_function):
        """
        Initializes a new instance of RecursiveCharacterTextSplitter
        
        :param chunk_size: Maximum size of chunks to return
        :param chunk_overlap: Overlap in characters between chunks
        :param length_function: Function that measures the length of given chunks
        
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function
        )
    
    def split_data(self, data):
        """
        Splits the given Document based on list of characters, ["\n\n", "\n", " ", ""]. Chunk size is measured of characters.
        
        :param data: The Document to be split, in the Document format returned by the langchain pdf loaders

        :return: Split Documents
        """
        docs = self.splitter.split_documents(data)
        return docs
