from langchain.vectorstores import ElasticsearchStore

from assistant.config import get_settings

from .embeddings import get_embeddings


def get_db():
    db = ElasticsearchStore(
        embedding=get_embeddings(),
        es_url=get_settings().ELASTIC_URL,
        es_api_key=get_settings().ELASTIC_KEY,
        index_name=get_settings().ELASTIC_INDEX,
    )

    return db
