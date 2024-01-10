import uuid
from types import NoneType
from typing import Any, Optional

import tenacity
from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct
from qdrant_client.models import Distance, VectorParams
from tenacity import retry_if_exception_type, wait_random_exponential

from lbgpt.semantic_cache.base import _SemanticCacheBase, get_completion_create_params
from lbgpt.types import ChatCompletionAddition


class QdrantSemanticCache(_SemanticCacheBase):
    ALLOWED_TYPES_FOR_PAYLOAD = (int, str, bool)

    def __init__(
        self,
        embedding_model: Embeddings,
        collection_name: str,
        host: Optional[str] = None,
        port: Optional[int] = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        qdrant_properties: Optional[dict[str, Any]] = None,
        cosine_similarity_threshold: float = 0.99,
    ):
        super().__init__(
            embedding_model=embedding_model,
            cosine_similarity_threshold=cosine_similarity_threshold,
        )

        # setting up the Qdrant client properly
        # we are setting up a synchroneous and an asynchroneous client
        # the synchroneous client is used for special operations like counting, listing, where we actually
        # care about precision

        self.qdrant_client = AsyncQdrantClient(
            host=host, port=port, url=url, api_key=api_key, **(qdrant_properties or {})
        )
        self._sync_qdrant_client = QdrantClient(
            host=host, port=port, url=url, api_key=api_key, **(qdrant_properties or {})
        )

        self.collection_name = collection_name

        collection_result = self._sync_qdrant_client.get_collections()

        existing_collection_names = [k.name for k in collection_result.collections]
        if collection_name not in existing_collection_names:
            # it is surprisingly difficult to figure out how long an embedding model is. Thus, we are actually
            # embedding a string here and then checking the length of the embedding
            _test_embedding = embedding_model.embed_query("test")

            self._sync_qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=len(_test_embedding), distance=Distance.COSINE
                ),
            )

    def _create_filter_value(
        self, value: int | str | bool | NoneType
    ) -> int | str | bool:
        if isinstance(value, self.ALLOWED_TYPES_FOR_PAYLOAD):
            return value
        elif value is None:
            return ""
        else:
            raise NotImplementedError(f"cannot process type {type(value)}")

    @tenacity.retry(
        retry=(retry_if_exception_type(ResponseHandlingException)),
        wait=wait_random_exponential(min=5, max=60),
    )
    async def query_cache(
        self, query: CompletionCreateParams | dict[str, Any]
    ) -> Optional[ChatCompletionAddition]:
        query = get_completion_create_params(**query)

        filter_params = self.non_message_dict(query)
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=k, match=MatchValue(value=self._create_filter_value(v))
                )
                for k, v in filter_params.items()
                if isinstance(v, self.ALLOWED_TYPES_FOR_PAYLOAD) or v is None
            ]
        )

        res = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self.embed_messages(query["messages"]),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=1,
            score_threshold=self.cosine_similarity_threshold,
        )

        res = [r for r in res if r.score >= self.cosine_similarity_threshold]

        if len(res) > 0:
            return ChatCompletionAddition(
                **res[0].payload["result"],
                is_exact=res[0].score >= 1 - 0.00001,
            )
        else:
            return


    @tenacity.retry(
        retry=(retry_if_exception_type(ResponseHandlingException)),
        wait=wait_random_exponential(min=5, max=60),
    )
    async def add_cache(
        self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        query = get_completion_create_params(**query)

        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self.embed_messages(query["messages"]),
                    payload={
                        "result": response.model_dump(
                            exclude={
                                "is_exact",
                            }
                        ),
                        **self.non_message_dict(
                            query,
                            allowed_types=self.ALLOWED_TYPES_FOR_PAYLOAD,
                            convert_not_allowed_to_empty=True,
                        ),
                    },
                )
            ],
        )

    @property
    def count(self) -> int:
        return self._sync_qdrant_client.count(
            collection_name=self.collection_name
        ).count
