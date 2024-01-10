import uuid
from typing import Any, Optional, List, Callable, Iterable
from app.embeddings.openai import OpenAI

from app.connectors.chroma import ChromaDBConnector

from app.core.config import settings, logger

default_collection_name: str = "default_collection"


class Chroma(object):

    def __init__(
            self,
            collection_name: str = settings.DEFAULT_COLLECTION_NAME,
            embedding_function: Optional[str] = None,
            persist_directory: Optional[str] = None,
            relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize with a Chroma client."""
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._relevance_score_fn = relevance_score_fn

        self.chroma_connector = ChromaDBConnector(host_url=settings.CHROMADB_CONNECTOR_SERVER_URL, jwt_token=settings.JWT_TOKEN)
        self._collection = None

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            if self._embedding_function == "openai":
                openai_obj = OpenAI()
                embeddings = openai_obj.get_len_safe_embeddings(texts)

        # chromadb_client = chromadb.HttpClient(
        #     host="localhost", port="8211", headers={"Authorization": "Bearer test-token"})
        # self._collection = self.chroma_connector.get_or_create_collection(self._collection_name)
        # os.environ["OPENAI_API_KEY"] = ""
        # openai_embeddings = OpenAIEmbeddings()
        # self._collection = chromadb_client.get_or_create_collection(self._collection_name,
        #                                                             embedding_function=openai_embeddings.embed_documents)

        self._collection = self.chroma_connector.get_or_create_collection(self._collection_name)

        print(self._collection)

        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    print(ids_with_metadata)
                    print(self._collection['id'])
                    self.chroma_connector.upsert_documents(collection_id=str(self._collection['id']),
                                                           ids=ids_with_metadata,
                                                           embeddings=embeddings_with_metadatas,
                                                           metadatas=metadatas,
                                                           documents=texts_with_metadatas)
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self.chroma_connector.upsert_documents(collection_id=str(self._collection['id']),
                                                       ids=ids_without_metadatas,
                                                       embeddings=embeddings_without_metadatas,
                                                       documents=texts_without_metadatas)
        else:
            self.chroma_connector.upsert_documents(collection_id=str(self._collection['id']), ids=ids,
                                                   embeddings=embeddings,
                                                   documents=texts)
        return ids
