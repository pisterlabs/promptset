import os
import chromadb
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings,HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import LongContextReorder
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.document_loaders import PyPDFLoader

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={"device":"cpu"},
                                      encode_kwargs = {'normalize_embeddings': False})
hf_bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                                             model_kwargs={"device":"cpu"},
                                             encode_kwargs = {'normalize_embeddings': False})
#Cohere/Cohere-embed-english-v3.0
hf_bge_base_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5",
                                             model_kwargs={"device":"cpu"},
                                             encode_kwargs = {'normalize_embeddings': False})



loader_mh  = PyPDFLoader("data/dall-e-3.pdf")
documnet_mh = loader_mh.load()
# print(len(documnet_mh))
loader_esops = PyPDFLoader("data/GPTV.pdf")
documnet_esops = loader_esops.load()
# print(len(documnet_esops))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_mh = text_splitter.split_documents(documnet_mh)
text_esops = text_splitter.split_documents(documnet_esops)

ABS_PATH = os.path.dirname(os.path.abspath("."))
DB_DIR = os.path.join(ABS_PATH, "db")
client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)

mh_vectorstore = Chroma.from_documents(text_mh,
                                       hf_bge_embeddings,
                                       client_settings=client_settings,
                                       collection_name="mh_dall_e",
                                       collection_metadata={"hnsw":"cosine"},
                                       persist_directory=DB_DIR)
esops_vectorstore = Chroma.from_documents(text_esops,
                                          hf_embeddings ,
                                          client_settings=client_settings,
                                          collection_name="esops_gptv",
                                          collection_metadata={"hnsw":"cosine"},
                                          persist_directory=DB_DIR)

retriever_mh = mh_vectorstore.as_retriever(search_type="mmr",
                                  search_kwargs={"k": 5, "include_metadata": True}
                                  )
retriever_esops = esops_vectorstore.as_retriever(search_type="mmr",
                                        search_kwargs={"k": 5, "include_metadata": True}
            )


lotr = MergerRetriever(retrievers=[retriever_mh, retriever_esops])

# for chunks in lotr.get_relevant_documents("What is GPT?"):
#     print(chunks.page_content)

filter = EmbeddingsRedundantFilter(embeddings=hf_bge_base_embeddings,)
pipeline = DocumentCompressorPipeline(transformers=[filter])
compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline,
                                                       base_retriever=lotr)

filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=hf_bge_base_embeddings,
    num_clusters=10,
    num_closest=1,
    sorted=True,
)

pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)

lotr = MergerRetriever(retrievers=[retriever_mh, retriever_esops])
query = "What is gpt?"
docs = lotr.get_relevant_documents(query)
# print(docs)

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
print(reordered_docs)