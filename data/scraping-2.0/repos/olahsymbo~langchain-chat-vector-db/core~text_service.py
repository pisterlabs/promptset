from langchain.document_loaders import TextLoader

from core.base_service import ChatService


class TextService(ChatService):

    def fetch_document(self):
        self.saved_file_path.endswith("txt")
        self.loaders = TextLoader(self.saved_file_path)
