import logging
from services.settings.settings import Settings
from services.qdrant.qdrant import QdrantClient
from services.ingest.utils import get_chunks
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings


class IngestClient:
    client = None
    vector_store = None
    _settings = Settings.get_settings()

    def __init__(self):
        self.qdrant_client = QdrantClient()
        self.qdrant_client.get_collection()
        self.__create_vector_store()

    def __create_vector_store(self):
        collection_name = self._settings.qdrant_collection_name
        open_ai_base_url = self._settings.open_ai_base
        if not open_ai_base_url:
            embeddings = OpenAIEmbeddings(
                openai_api_key=self._settings.openai_api_key,
            )
        else:
            embeddings = OpenAIEmbeddings(
                openai_api_key=self._settings.openai_api_key,
                openai_api_base=open_ai_base_url,
                deployment=self._settings.open_ai_deployment_name,
                api_version=self._settings.open_ai_api_version,
                openai_api_type=self._settings.open_ai_type,
                model=self._settings.open_ai_model_name,
            )
        self.vector_store = Qdrant(
            client=self.qdrant_client.client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

    def ingest_document(self, pdf_body: dict, file_metadata: dict) -> bool:
        """Ingest document"""
        full_text = ""
        try:
            for _, page_data in pdf_body.items():
                full_text += page_data
            text_chunks = get_chunks(full_text)
            self.vector_store.add_texts(
                texts=text_chunks,
                metadatas=[file_metadata] * len(text_chunks),
            )
            return True
        except Exception as e:
            logging.exception(f"[LLM Ingest] {e}")
            return False
