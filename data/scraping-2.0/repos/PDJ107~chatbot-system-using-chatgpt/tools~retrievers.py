from langchain.retrievers import ElasticSearchBM25Retriever
from langchain.schema import Document
from typing import List, Any


class CustomRetriever(ElasticSearchBM25Retriever):
    def __init__(self, client: Any, index_name: str, k=1):
        super().__init__(client, index_name)
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        q = []
        for qq in query.split(' '):
            if qq not in ['한기대', '위치', '일정']:
                q.append(qq)
        query = ' '.join(q)
        query_dict = {"multi_match": {"query": query, "fields": ['*']}}
        res = self.client.search(index=self.index_name, query=query_dict, size=self.k)

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(
                Document(
                    page_content=r["_source"]["page_content"],
                    metadata=r["_source"]['metadata']
                )
            )
        # print(docs)
        return docs
