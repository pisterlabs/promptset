from langchain.document_loaders import Docx2txtLoader

from core.base_service import ChatService


class DocService(ChatService):

    def fetch_document(self):
        self.saved_file_path.endswith("docx")
        self.loaders = Docx2txtLoader(self.saved_file_path)

