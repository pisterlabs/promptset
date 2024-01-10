import streamlit as st
import llama_index
from llama_index import StorageContext, load_index_from_storage
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index import KeywordTableIndex
from llama_index.indices.keyword_table import SimpleKeywordTableIndex
from llama_index import ResponseSynthesizer
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.retrievers import VectorIndexRetriever
from llama_index.retrievers import ListIndexRetriever
from llama_index.retrievers import TreeRootRetriever
from llama_index.indices.keyword_table.retrievers import KeywordTableGPTRetriever
from llama_index.indices.keyword_table import GPTSimpleKeywordTableIndex
from llama_index.indices.keyword_table.retrievers import KeywordTableRAKERetriever
from llama_index.indices.keyword_table.retrievers import KeywordTableSimpleRetriever
from llama_index import Prompt
from llama_index import LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext


print("1")
storage_context_1 = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector_store"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="vector_store"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector_store"),
)

storage_context_2 = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="table"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="table"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="table"),
)
storage_context_3 = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="tree"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="tree"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="tree"),
)
storage_context_4 = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="list"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="list"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="list"),
)

print("2")
from llama_index import load_index_from_storage, load_indices_from_storage, load_graph_from_storage

indices1 = load_index_from_storage(storage_context_1)
indices2 = load_index_from_storage(storage_context_2)
indices3 = load_index_from_storage(storage_context_3)
indices4 = load_index_from_storage(storage_context_4)
# indices1 = load_index_from_storage(storage_context="vector_store")

index = [indices1, indices2, indices3, indices4]

print("3")

print("4")
from llama_index.indices.response import BaseResponseBuilder





# configure response synthesizer
response_synthesizer = ResponseSynthesizer.from_args(
    # node_postprocessors=[
    # ]
)


    
print("5")


TEMPLATE_STR = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
QA_TEMPLATE = Prompt(TEMPLATE_STR)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=1024)




query_engine1 = indices3.as_query_engine(service_context=service_context, text_qa_template=QA_TEMPLATE, similarity_top_k=3, streaming=True, )


response = query_engine1.query('How much package has government of india announced?')


# print("7")
str(response)
print(response)


# response.source_nodes
print(response.source_nodes)


########## working ##########