from typing import Optional, Any, List, Tuple, Callable

from langchain.schema import Document
from langchain.vectorstores import Clickhouse


class ClickhousePro(Clickhouse):

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    def _build_query_sql(self, q_emb: List[float], topk: int, where_str: Optional[str] = None) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        settings_strs = []
        if self.config.index_query_params:
            for k in self.config.index_query_params:
                settings_strs.append(f"SETTING {k}={self.config.index_query_params[k]}")
        q_str = f"""
                    SELECT {self.config.column_map['document']}, 
                        {self.config.column_map['metadata']}, dist
                    FROM {self.config.database}.{self.config.table}
                    {where_str}
                    ORDER BY cosineDistance({self.config.column_map['embedding']}, [{q_emb_str}]) 
                        AS dist {self.dist_order}
                    LIMIT {topk} {' '.join(settings_strs)}
                    """
        return q_str

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4, where_str: Optional[str] = None,
                                                **kwargs: Any) -> List[Tuple[Document, float]]:
        q_str = self._build_query_sql(
            self.embedding_function.embed_query(query), k, where_str
        )
        docs_and_dist = [
            (
                Document(
                    page_content=r[self.config.column_map["document"]],
                    metadata=r[self.config.column_map["metadata"]],
                ),
                r["dist"],
            )
            for r in self.client.query(q_str).named_results()
        ]
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_similarities = [(doc, relevance_score_fn(score)) for doc, score in docs_and_dist]
        score_threshold = kwargs.pop("score_threshold", None)
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
        return docs_and_similarities


