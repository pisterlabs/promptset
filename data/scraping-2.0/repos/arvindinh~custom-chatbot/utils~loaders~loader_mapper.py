from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

class LoaderMapper:
    """
    LoaderMapper can accept multiple file types and return a langchain loader wrapper that corresponds to the associated loader.
    Currently supports csv, pdf, txt, html, md, docx, pptx, xls, xlsx.

    Note: Currently having issues with JSON
    """
    #keep dict of file extensions and their relevant loaders with their arguments
    loader_map = {
            ".csv": (CSVLoader, {}),
            ".pdf": (PyMuPDFLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".xls": (UnstructuredExcelLoader, {}),
            ".xlsx": (UnstructuredExcelLoader, {}),
        }
    
    @classmethod
    def find_loader(self, filepath):
        """
        Finds the associated loader based on filepath extension
        
        :param filepath: path of the file to be loaded
        
        :return: langchain loader wrapper object. to load the filepath into a Document object, use ".load"

        Example usage:
            mapper = LoaderMapper()
            loader = mapper.find_loader(filepath)
            data = loader.load()
        
        You can pass in the data(Document object) to a splitter, which returns the chunks you can pass to create an embedding/store in db
        """
        ext = "." + filepath.rsplit(".", 1)[-1]
        if ext in LoaderMapper.loader_map:
            loader_class, loader_args = LoaderMapper.loader_map[ext]
            loader = loader_class(filepath, **loader_args)
            return loader
        
        raise ValueError(f"Unsupported file extension '{ext}'")
