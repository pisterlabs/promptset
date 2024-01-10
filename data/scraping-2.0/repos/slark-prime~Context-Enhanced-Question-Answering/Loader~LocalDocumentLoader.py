from Loader.BaseLoader import *
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.docstore.document import Document


class LocalDocumentLoader(BaseLoader):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path


    def load_document(self):
        if self.file_path.endswith(".txt"):
            loader = TextLoader(self.file_path, encoding="utf8")
        elif self.file_path.endswith(".pdf"):
            loader = PDFMinerLoader(self.file_path)
        elif self.file_path.endswith(".csv"):
            loader = CSVLoader(self.file_path)

        self.data = loader.load()[0]

    def preprocess_data(self) -> Document:
        self.data.page_content = clean_string(self.data.page_content)

    def load(self):
        """Load and preprocess data from the source.

        Returns:
            The loaded and preprocessed data.
        """
        self.load_document()
        self.preprocess_data()

        return self.data