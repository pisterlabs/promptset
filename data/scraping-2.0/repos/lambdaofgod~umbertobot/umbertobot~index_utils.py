from typing import Optional

import qdrant_client
from langchain.vectorstores import Qdrant
from qdrant_client.http import models as qdrant_models


def get_default_qdrant_client():
    url: Optional[str] = None
    port: Optional[int] = 6333
    grpc_port: int = 6334
    return qdrant_client.QdrantClient(
        host="localhost", url=url, port=port, grpc_port=grpc_port
    )


def update_qdrant_alias(
    qdrant,
    collection_name,
):
    collection_name = list(qdrant._client.collections)[0]
    collection_name
    qdrant.update_collection_aliases(
        [
            qdrant_models.CreateAliasOperation(
                create_alias=qdrant_models.CreateAlias(
                    collection_name=collection_name, alias_name=collection_name
                )
            )
        ]
    )
