""" Document loader functions """
from langchain.document_loaders import (
    UnstructuredExcelLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
# define a function to load documents
def load_documents(files, file_types):
    """ Load documents from a list of file paths. """
    # Initialize the document loader
    document_loader = None
    documents = []
    # Iterate over the files
    for file, file_type in zip(files, file_types):
        # Load the documents
        if file_type == 'csv':
            document_loader = CSVLoader(file)
        elif file_type == 'pdf':
            document_loader = PyPDFLoader(file)
        elif file_type == 'txt':
            document_loader = TextLoader(file)
        elif file_type == 'xlsx':
            document_loader = UnstructuredExcelLoader(file)
        else:
            raise Exception(f'File type {file_type} not supported.')
        # Append the documents to the list
        documents.append(document_loader.load())
    # Return the documents
    return documents