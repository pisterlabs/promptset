from langchain_core.embeddings import Embeddings
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.vectorstores import VectorStore

from app.settings import settings


class RepoLoader:
    def __init__(
        self,
        path: str,
        embedding_model: Embeddings,
        vector_store: VectorStore,
    ):
        self.path = path
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.loader = GenericLoader.from_filesystem(
            self.path,
            suffixes=settings.loader_suffixes,
            exclude=settings.loader_exclude_paths,
            show_progress=True,
            parser=LanguageParser(
                language=settings.loader_language, parser_threshold=500
            ),
        )
        print(self.loader.__dict__)

    def load(self):
        documents = self.loader.load()
        print(f"Loaded {len(documents)} documents")
        return self.vector_store.add_documents(documents)
