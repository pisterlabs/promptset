from langchain.text_splitter import RecursiveCharacterTextSplitter

from strategy_ai.ai_core.document_loaders.general_loader import DocumentSource


class DocStore():
    def __init__(self, documentSources: dict[str, DocumentSource]):
        self.documents = []
        self.splitDocuments = []

        for documentSource in documentSources.values():
            self.documents.extend(documentSource.documents)

        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"]
        )
        self.splitDocuments.extend(
            self.textSplitter.split_documents(self.documents))
