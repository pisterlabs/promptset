from llama_index import StorageContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import LLMPredictor
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.logger.base import LlamaLogger
from llama_index.callbacks.base import CallbackManager


import json

from llama_index import ListIndex
from llama_index import SimpleDirectoryReader

# ------------------------- Storage Context -------------------------
# SimpleDirectoryReader は InMemory なストア。
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# Index: https://gpt-index.readthedocs.io/en/stable/module_guides/indexing/index_guide.html
list_index = ListIndex.from_documents(documents)

# document store の dump.
with open("tmp/docstore.json", "wt") as f:
    json.dump(list_index.storage_context.docstore.to_dict(), f, indent=4)

# vector store の dump.
with open("tmp/vector_store.json", "wt") as f:
    json.dump(list_index.storage_context.vector_store.to_dict(), f, indent=4)

# document store の詳細。
for doc_id, node in list_index.storage_context.docstore.docs.items():
    node_dict = node.__dict__
    print(f'{doc_id=}, len={len(node_dict["text"])}, start={node_dict["start_char_idx"]}, end={node_dict["end_char_idx"]}')


# ------------------------- Service Context -------------------------
print(list_index.service_context.__dict__)

# embed_model の詳細 (OpenAIEmbedding)。
# {
#   'model_name': 'text-embedding-ada-002',
#  ...
print(list_index.service_context.embed_model.__dict__)


# ------------------------- Query Engine -------------------------
query_engine = list_index.as_query_engine()
print(type(query_engine))
print(query_engine)
# {'_retriever': <llama_index.indices.list.retrievers.SummaryIndexRetriever ...
print(query_engine.__dict__)
