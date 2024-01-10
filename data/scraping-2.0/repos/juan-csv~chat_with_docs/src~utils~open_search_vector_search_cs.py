from langchain.schema.embeddings import Embeddings
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch, _approximate_search_query_with_boolean_filter, _approximate_search_query_with_efficient_filter,
    _default_approximate_search_query, _default_script_query, _default_painless_scripting_query,
    SCRIPT_SCORING_SEARCH,MATCH_ALL_QUERY,PAINLESS_SCRIPTING_SEARCH

)

class OpenSearchVectorSearchCS(OpenSearchVectorSearch):

    def __init__(self, opensearch_url: str, index_name: str, embedding_function: Embeddings, **kwargs: Any):
        super().__init__(opensearch_url, index_name, embedding_function, **kwargs)

    def get_kwargs_value(self, kwargs: Any, key: str, default_value: Any) -> Any:
        """Get the value of the key if present. Else get the default_value."""
        if 'kwargs' in kwargs.keys():
            if key in kwargs['kwargs']:
                return kwargs['kwargs'].get(key)
            else:
                return default_value
        elif key in kwargs:
            return kwargs.get(key)
        else:
            return default_value

    def _raw_similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[dict]:
        """Return raw opensearch documents (dict) including vectors,
        scores most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of dict with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        embedding = self.embedding_function.embed_query(query)
        search_type = self.get_kwargs_value(kwargs, "search_type", "approximate_search")
        vector_field = self.get_kwargs_value(kwargs, "vector_field", "vector_field")
        index_name = self.get_kwargs_value(kwargs, "index_name", self.index_name)

        if (
            self.is_aoss
            and search_type != "approximate_search"
            and search_type != SCRIPT_SCORING_SEARCH
        ):
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search` and `script_scoring`"
            )

        if search_type == "approximate_search":
            boolean_filter = self.get_kwargs_value(kwargs, "boolean_filter", {})
            subquery_clause = self.get_kwargs_value(kwargs, "subquery_clause", "must")
            efficient_filter = self.get_kwargs_value(kwargs, "efficient_filter", {})
            # `lucene_filter` is deprecated, added for Backwards Compatibility
            lucene_filter = self.get_kwargs_value(kwargs, "lucene_filter", {})

            if boolean_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `boolean_filter` and `efficient_filter` are provided which "
                    "is invalid"
                )

            if lucene_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `efficient_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if lucene_filter != {} and boolean_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `boolean_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(
                    embedding,
                    boolean_filter,
                    k=k,
                    vector_field=vector_field,
                    subquery_clause=subquery_clause,
                )
            elif efficient_filter != {}:
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding, efficient_filter, k=k, vector_field=vector_field
                )
            elif lucene_filter != {}:
                warnings.warn(
                    "`lucene_filter` is deprecated. Please use the keyword argument"
                    " `efficient_filter`"
                )
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding, lucene_filter, k=k, vector_field=vector_field
                )
            else:
                search_query = _default_approximate_search_query(
                    embedding, k=k, vector_field=vector_field
                )
        elif search_type == SCRIPT_SCORING_SEARCH:
            space_type = self.get_kwargs_value(kwargs, "space_type", "l2")
            pre_filter = self.get_kwargs_value(kwargs, "pre_filter", MATCH_ALL_QUERY)
            search_query = _default_script_query(
                embedding, k, space_type, pre_filter, vector_field
            )
        elif search_type == PAINLESS_SCRIPTING_SEARCH:
            space_type = self.get_kwargs_value(kwargs, "space_type", "l2Squared")
            pre_filter = self.get_kwargs_value(kwargs, "pre_filter", MATCH_ALL_QUERY)
            search_query = _default_painless_scripting_query(
                embedding, k, space_type, pre_filter, vector_field
            )
        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        response = self.client.search(index=index_name, body=search_query)

        return [hit for hit in response["hits"]["hits"]]