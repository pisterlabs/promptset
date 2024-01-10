from langchain.document_loaders import PyPDFLoader
from core.base_service import ChatService


class PDFService(ChatService):

    def fetch_document(self):
        self.saved_file_path.endswith("pdf")
        self.loaders = PyPDFLoader(self.saved_file_path)
