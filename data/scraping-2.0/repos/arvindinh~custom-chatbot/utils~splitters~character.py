from langchain.text_splitter import CharacterTextSplitter

class Character_TextSplitter:
    """
    A class to split a Document using the CharacterTextSplitter wrapper from the langchain library
    """
    def __init__(self, separator, chunk_size, chunk_overlap, length_function):
        """
        Initializes a new instance of CharacterTextSplitter
        ex. 
            splitter = character.Character_TextSplitter(
                separator= "\n",
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len,
            )
        :param separator: list of separator characters for the text splitter
        :param chunk_size: Maximum size of chunks to return
        :param chunk_overlap: Overlap in characters between chunks
        :param length_function: Function that measures the length of given chunks
        
        """
        self.splitter = CharacterTextSplitter(
            separator = separator,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function
        )
    
    def split_data(self, data):
        """
        Splits the given Document based on single characters, default "\n\n", and measures chunk length by number of characters
        
        :param data: The Document to be split, in the Document format returned by the langchain pdf loaders

        :return: Split Documents
        """
        docs = self.splitter.split_documents(data)
        return docs
