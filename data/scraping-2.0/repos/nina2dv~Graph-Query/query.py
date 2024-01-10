from llama_index import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)

from llama_index.storage.storage_context import StorageContext
from dotenv import load_dotenv
import os
from llama_index.graph_stores import SimpleGraphStore
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import Cohere
from llama_index.llms import Ollama
from IPython.display import Markdown, display

from llama_index.llms import HuggingFaceInferenceAPI
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from IPython.display import Markdown, display
from llama_index import download_loader

from llama_index import QueryBundle
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
)

from typing import List
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
from langchain.embeddings import OllamaEmbeddings

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
# documents = SimpleDirectoryReader("/notes").load_data()
documents = loader.load_data(pages=["2023 in science"], auto_suggest=False)
# print(len(documents))

# Set llm
load_dotenv()
HF_TOKEN = os.environ.get('HF_KEY')
llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN
)
# llm = Ollama(model="llama2")

cohere_api_key = os.environ.get('COHERE_API_KEY')
# model = "command"
# temperature = 0
# max_tokens = 256
# llm = Cohere(model=model,temperature=0,api_key=cohere_api_key,max_tokens=max_tokens)

# with input_typ='search_query'
# embed_model = CohereEmbedding(
#             cohere_api_key=cohere_api_key,
#             model_name="embed-english-v3.0",
#             input_type="search_query",
#             embed_batch_size=42
#         )
embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="thenlper/gte-large")
)
# embed_model = OllamaEmbeddings()
# embed_model = LangchainEmbedding(
#   HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# )

# setup the service context
service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

# set the storage context
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Construct the Graph Index
kg_index = KnowledgeGraphIndex.from_documents(documents=documents,
                                              max_triplets_per_chunk=2,
                                              service_context=service_context,
                                              storage_context=storage_context,
                                              include_embeddings=True)

# vector_index = VectorStoreIndex.from_documents(documents)
#
#
# class CustomRetriever(BaseRetriever):
#     """Custom retriever that performs both Vector search and Knowledge Graph search"""
#
#     def __init__(
#             self,
#             vector_retriever: VectorIndexRetriever,
#             kg_retriever: KGTableRetriever,
#             mode: str = "OR",
#     ) -> None:
#         """Init params."""
#
#         self._vector_retriever = vector_retriever
#         self._kg_retriever = kg_retriever
#         if mode not in ("AND", "OR"):
#             raise ValueError("Invalid mode.")
#         self._mode = mode
#         super().__init__()
#
#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""
#
#         vector_nodes = self._vector_retriever.retrieve(query_bundle)
#         kg_nodes = self._kg_retriever.retrieve(query_bundle)
#
#         vector_ids = {n.node.node_id for n in vector_nodes}
#         kg_ids = {n.node.node_id for n in kg_nodes}
#
#         combined_dict = {n.node.node_id: n for n in vector_nodes}
#         combined_dict.update({n.node.node_id: n for n in kg_nodes})
#
#         if self._mode == "AND":
#             retrieve_ids = vector_ids.intersection(kg_ids)
#         else:
#             retrieve_ids = vector_ids.union(kg_ids)
#
#         retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
#         return retrieve_nodes
#
#
# # create custom retriever
# vector_retriever = VectorIndexRetriever(index=vector_index)
# kg_retriever = KGTableRetriever(
#     index=kg_index, retriever_mode="keyword", include_text=False
# )
# custom_retriever = CustomRetriever(vector_retriever, kg_retriever)
#
# # create response synthesizer
# response_synthesizer = get_response_synthesizer(
#     service_context=service_context,
#     response_mode="tree_summarize",
# )
#
# custom_query_engine = RetrieverQueryEngine(
#     retriever=custom_retriever,
#     response_synthesizer=response_synthesizer,
# )
#
# vector_query_engine = vector_index.as_query_engine()
#
# kg_keyword_query_engine = kg_index.as_query_engine(
#     # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
#     include_text=False,
#     retriever_mode="keyword",
#     response_mode="tree_summarize",
#     # embedding_mode="hybrid",
#     similarity_top_k=5
# )
#
# response = kg_keyword_query_engine.query("Tell me interesting science events in 2023")
# print(response)

query = "Tell me interesting science events in 2023"
query_engine = kg_index.as_query_engine(
 include_text=True,
 response_mode="tree_summarize",
 embedding_mode="hybrid",
 similarity_top_k=5,
)

response = query_engine.query(query)
display(Markdown(f"<b>{response}</b>"))
