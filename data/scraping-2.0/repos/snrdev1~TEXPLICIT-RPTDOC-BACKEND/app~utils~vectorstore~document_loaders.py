from langchain.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


def load_pdf(filepath):
    """
    The function `load_pdf` loads an unstructured PDF file and returns the loaded data.
    
    Args:
      filepath: The filepath parameter is a string that represents the path to the PDF file that you
    want to load.
    
    Returns:
      the data loaded from the PDF file.
    """
    loader = UnstructuredPDFLoader(filepath)
    data = loader.load()

    return data


def load_docx(filepath):
    """
    The function `load_docx` loads and returns the data from a Word document specified by the `filepath`
    parameter.
    
    Args:
      filepath: The filepath parameter is a string that represents the path to the .docx file that you
    want to load.
    
    Returns:
      the data loaded from the Word document.
    """
    loader = UnstructuredWordDocumentLoader(filepath)
    data = loader.load()

    return data


def load_pptx(filepath):
    """
    The function `load_pptx` loads an unstructured PowerPoint file and returns the loaded data.
    
    Args:
      filepath: The filepath parameter is a string that represents the path to the PowerPoint file that
    you want to load.
    
    Returns:
      the data loaded from the PowerPoint file.
    """
    loader = UnstructuredPowerPointLoader(filepath)
    data = loader.load()

    return data


def load_txt(filepath):
    """
    The function `load_txt` loads text from a file using a `TextLoader` object and returns the loaded
    pages.
    
    Args:
      filepath: The filepath parameter is a string that represents the path to the text file that you
    want to load.
    
    Returns:
      the pages loaded from the text file.
    """
    loader = TextLoader(filepath)
    pages = loader.load()

    return pages
