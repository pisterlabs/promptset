import os

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Redis
from llm.llm import llm_embedding, llm


class SearchDocument:
    redis_url = os.getenv("REDIS_URL", "redis://4.236.196.174:6379")
    count = os.getenv("REDIS_SEARCH_COUNT", 2)

    @classmethod
    def search_document(cls, question, index_name):
        redis = Redis.from_existing_index(embedding=llm_embedding, index_name=index_name, redis_url=cls.redis_url)
        docs = redis.similarity_search(query=question, k=cls.count)
        print(docs)
        print(type(docs))
        chain = load_qa_with_sources_chain(llm=llm, verbose=True)
        return chain({"input_documents": docs, "question": question}, return_only_outputs=True)
