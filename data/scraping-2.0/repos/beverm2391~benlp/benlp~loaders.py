from langchain.document_loaders import \
    PyPDFLoader, UnstructuredExcelLoader, NotebookLoader as langchain_NotebookLoader, \
    Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.document_loaders.csv_loader import CSVLoader as langchain_CSVLoader

import os
import subprocess
import sys


class PDFLoader(PyPDFLoader):
    """
    A wrapper around langchain.document_loaders.PyPDFLoader, for future extensibility.
    """
    try:
        import pypdf
    except ImportError:
        bool_ = input("pypdf not installed, Would you like to install it now? (y/n)")
        if bool_ == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])

    def __init__(self, fpath):
        super().__init__(fpath)


class CSVLoader(langchain_CSVLoader):
    """
    A wrapper around langchain.document_loaders.CSVLoader, for future extensibility.
    """

    def __init__(self, fpath):
        super().__init__(fpath)


class ExcelLoader(UnstructuredExcelLoader):
    """
    A wrapper around langchain.document_loaders.UnstructuredExcelLoader, for future extensibility.
    """

    def __init__(self, fpath):
        super().__init__(fpath)


class NotebookLoader(langchain_NotebookLoader):
    """
    A wrapper around langchain.document_loaders.NotebookLoader, for future extensibility.
    """
    try:
        import pandas
    except ImportError:
        bool_ = input("pandas not installed, Would you like to install it now? (y/n)")
        if bool_ == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

    def __init__(self, fpath):
        super().__init__(fpath)


class DocxLoader(Docx2txtLoader):
    """
    A wrapper around langchain.document_loaders.Docx2txtLoader, for future extensibility.
    """

    def __init__(self, fpath):
        super().__init__(fpath)

class PptxLoader(UnstructuredPowerPointLoader):
    """
    A wrapper around langchain.document_loaders.UnstructuredPowerPointLoader, for future extensibility.
    """

    def __init__(self, fpath):
        super().__init__(fpath)

# ! Factory Function ================================================================

def create_loader(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Document at {fpath} does not exist.")
    
    _, ext = os.path.splitext(fpath)
    ext = ext.lower()

    if ext == '.pdf':
        return PDFLoader(fpath)
    elif ext == '.csv':
        return CSVLoader(fpath)
    elif ext in ['.xls', '.xlsx']:
        return ExcelLoader(fpath)
    elif ext == '.ipynb':
        return NotebookLoader(fpath)
    elif ext == '.docx':
        return DocxLoader(fpath)
    elif ext == '.pptx':
        return PptxLoader(fpath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")