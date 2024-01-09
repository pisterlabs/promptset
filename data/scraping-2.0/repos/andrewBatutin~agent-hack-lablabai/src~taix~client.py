import os
from typing import Any, Dict, List

import weaviate
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import Weaviate

from src.schema import DOC_CLASS, TAX_LIMIT

load_dotenv()


class MyWeaviate(Weaviate):
    def similarity_search_by_text(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        content: Dict[str, Any] = {"concepts": [query]}
        if kwargs.get("search_distance"):
            content["certainty"] = kwargs.get("search_distance")
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))
        if kwargs.get("additional"):
            query_obj = query_obj.with_additional(kwargs.get("additional"))
        result = query_obj.with_near_text(content).with_limit(k).do()
        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")
        docs = []
        for res in result["data"]["Get"][self._index_name]:
            # text = res.pop(self._text_key)
            # we need more than just the _text_key for valid answer
            text = str(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs


def get_batch_with_cursor(client, class_name, batch_size=20, class_properties=None, filter=None, cursor=None):
    query = client.query.get(class_name, class_properties).with_where(filter).with_limit(batch_size)

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()


def get_all_docs(features, filter):
    client = weaviate.Client(
        "http://localhost:8080",
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WV_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    docs = get_batch_with_cursor(
        client,
        DOC_CLASS,
        class_properties=features,
        filter=filter,
    )
    return docs


def wv_retriever(w_url):
    client = weaviate.Client(
        w_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WV_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    vectorstore = MyWeaviate(
        client,
        DOC_CLASS,
        "invoice_items",
        attributes=["invoice_number", "value", "currency", "recipient_address", "country"],
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": 20}
        # search_kwargs={"k": 20, "where_filter": {"path": ["country"], "operator": "NotEqual", "valueText": "Germany"}}
    )


def wv_retriever_limits(w_url):
    client = weaviate.Client(
        w_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WV_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    vectorstore = MyWeaviate(
        client,
        TAX_LIMIT,
        "limit_value",
        attributes=["rule", "currency"],
    )
    return vectorstore.as_retriever(search_kwargs={"k": 1})
