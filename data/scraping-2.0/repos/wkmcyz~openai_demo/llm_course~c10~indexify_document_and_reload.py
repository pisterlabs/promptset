import openai
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.indices.base import BaseGPTIndex

from util.proxy import set_proxy


def load_local_mr_fujino_index() -> BaseGPTIndex:
    # 从磁盘重新加载：
    from llama_index import StorageContext, load_index_from_storage

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./index_mr_fujino")
    # load index
    index = load_index_from_storage(storage_context)
    return index


def indexify_document():
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    # save
    index.storage_context.persist('index_mr_fujino')


if __name__ == '__main__':
    set_proxy()
    index = load_local_mr_fujino_index()
    query_engine = index.as_query_engine()
    # response = query_engine.query("鲁迅先生在日本学习医学的老师是谁？")
    # 鲁迅先生去了仙台。
    response = query_engine.query("鲁迅先生去了哪里？")
    print(response)
