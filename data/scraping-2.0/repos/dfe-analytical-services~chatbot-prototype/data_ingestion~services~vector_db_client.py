import logging
from itertools import chain

import openai
import qdrant_client.models as models
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import settings
from ..utils.text_utils import chunk_text

logger = logging.getLogger(__name__)

client = QdrantClient(location=settings.qdrant_host, port=settings.qdrant_port)


def upsert_data(records: list[dict[str, str]], batch_size: int = 100) -> None:
    ensure_collection_exists()

    chunks = list(chain.from_iterable(map(create_url_text_map, records)))

    for i in range(0, len(chunks), batch_size):
        end_index = min(i + batch_size, len(chunks))
        batch_meta = chunks[i:end_index]
        batch_text = [chunks[j]["text"] for j in range(i, end_index)]
        try:
            embeds = openai.Embedding.create(input=batch_text, engine=settings.openai_embedding_model)
        except Exception as e:
            logger.error(f"An error occured within embedding model: {e}")

        formatted_embeddings = [embeds["data"][j]["embedding"] for j in range(len(embeds["data"]))]

        client.upsert(
            collection_name=settings.qdrant_collection,
            points=models.Batch(
                ids=[j for j in range(i, end_index)],
                payloads=batch_meta,
                vectors=formatted_embeddings,
            ),
        )

        logger.debug("Batch upserted")

    logger.info("Data upsertion completed")


def create_url_text_map(record: dict[str, str]) -> list[dict[str, str]]:
    if record is None:
        return []
    chunks = chunk_text(text=record["text"])
    return list(map(lambda text: {"url": record["link"], "text": text}, chunks))


def recreate_collection() -> bool:
    return client.recreate_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(distance=models.Distance.COSINE, size=1536),
    )


def ensure_collection_exists() -> None:
    try:
        client.get_collection(collection_name=settings.qdrant_collection)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.debug("Collection doesn't exist - recreating collection")
            recreate_collection()
        else:
            raise


def delete_url(url: str) -> None:
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=url),
                    ),
                ]
            ),
        ),
    )
