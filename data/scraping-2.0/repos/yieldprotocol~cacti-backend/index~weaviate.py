from typing import Optional, List
import uuid

import weaviate  # type: ignore
from langchain.vectorstores import Weaviate

import utils
import registry


# set an arbitrary uuid for namespace, for consistent uuids for objects
NAMESPACE_UUID = uuid.UUID('64265e01-0339-4063-8aa3-bcd562b55aea')

auth_config = weaviate.AuthApiKey(api_key=utils.WEAVIATE_API_KEY)

def get_client() -> weaviate.Client:
    client = weaviate.Client(
        url=utils.WEAVIATE_URL,
        auth_client_secret=auth_config if utils.WEAVIATE_API_KEY else None,
        additional_headers={"X-OpenAI-Api-Key": utils.OPENAI_API_KEY},
    )
    return client


@registry.register_class
class WeaviateIndex(Weaviate):
    """Thin wrapper around langchain's vector store."""
    def __init__(self, index_name: str, text_key: str, extra_keys: Optional[List[str]] = None) -> None:
        client = get_client()
        super().__init__(client, index_name, text_key, attributes=extra_keys or [])
