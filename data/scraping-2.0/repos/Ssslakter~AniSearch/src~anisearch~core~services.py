from operator import itemgetter
from qdrant_client.models import Filter, FieldCondition, Record, MatchValue, MatchText, Range
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from src.features.document_builder import DocumentBuilder
from src.anisearch.configs import Config
from src.anisearch.core.storage import Storage
from src.anisearch.data_models import AnimeData
from src.anisearch.core.recsys import RecSys


class Services:
    """Class that holds all services"""

    OUTPUT_AMOUNT = 15

    filter = Filter(
        must=[
            FieldCondition(
                key="metadata.score",
                range=Range(gt=5.3),
            ),
            FieldCondition(
                key="metadata.popularity",
                range=Range(lt=7000),
            ),
        ]
    )

    def __init__(self, config: Config) -> None:
        self.embeddings = get_embeddings(config.models_dir)
        self.config = config
        self.document_builder = DocumentBuilder()
        self.storage = Storage(self.embeddings, self.config.qdrant)
        self.recsys = RecSys(config.recsys_dir)

    def initialize(self):
        """Perform some init configurations like creating collections, etc."""
        self.storage.init_collection(self.config.qdrant.collection_name)

    def insert_anime(self, anime: AnimeData) -> list[str]:
        """Inserts anime into storage"""
        doc = self.document_builder.render_document(anime)
        return self.storage.qdrant.add_documents([doc])

    async def search_anime(self, query: str) -> list[Document]:
        docs = self.qdrant_search(query, k=200)

        return list(docs)

    async def full_search(self, query: str) -> list[Document]:
        docs = self.fulltext_search(query, k=200)

        return list(docs)

    def delete_anime(self, uid: int):
        self.storage.qdrant.client.delete(
            self.config.qdrant.collection_name,
            points_selector=Filter(
                should=[FieldCondition(key="metadata.uid", match=MatchValue(value=uid))]
            ),
        )

    def get_anime(self, uid: int):
        response = self.storage.qdrant.client.scroll(
            self.config.qdrant.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.uid", match=MatchValue(value=uid))]
            ),
            limit=1,
        )[0]
        if not response:
            return None
        return self._to_document(response[0])

    def recommend_anime(self, nickname: str):
        uids = self.recsys.recommend(nickname, self.OUTPUT_AMOUNT)
        return [self.get_anime(uid) for uid in uids]

    async def recommend_from_relevant(self, nickname: str, query: str):
        animes = await self.search_anime(query)
        return self.recsys.rec_from_relevant(nickname, animes)

    def get_collections(self):
        return self.storage.client.get_collections()

    def _to_document(self, point: Record):
        return Document(
                page_content=point.payload["page_content"],  # type:ignore
                metadata=point.payload["metadata"],  # type:ignore
            )

    def qdrant_search(self, query: str, k: int = 4) -> list[Document]:
        results = self.storage.client.search(
            collection_name=self.config.qdrant.collection_name,
            query_vector=self.storage.qdrant._embed_query(query),
            query_filter=self.filter,
            with_payload=True,
            limit=k,
        )
        results = [
            (
                self.storage.qdrant._document_from_scored_point(
                    result,
                    self.storage.qdrant.content_payload_key,
                    self.storage.qdrant.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]
        return list(map(itemgetter(0), results))

    def fulltext_search(self, query: str, k: int = 4) -> list[Document]:
        data = self.storage.qdrant.client.scroll(
            self.config.qdrant.collection_name,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="page_content", match=MatchText(text=query.lower())),
                ]
            ),
            limit=k
        )[0]

        return [
            self.storage.qdrant._document_from_scored_point(
                result,
                self.storage.qdrant.content_payload_key,
                self.storage.qdrant.metadata_payload_key,
            )
            for result in data
        ]


def get_embeddings(models_dir: str) -> HuggingFaceEmbeddings:
    """Get default huggingface embeddings from models directory"""

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=models_dir,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
