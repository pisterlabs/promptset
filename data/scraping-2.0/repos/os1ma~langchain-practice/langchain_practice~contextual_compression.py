from langchain.document_loaders import TextLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
    LLMChainFilter,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from util import initialize, pretty_print_docs

initialize()

# Using a vanilla vector store retriever
documents = TextLoader("./docs/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

# docs = retriever.get_relevant_documents(
#     "What did the president say about Ketanji Brown Jackson"
# )
# pretty_print_docs(docs)

# Adding contextual compression with an LLMChainExtractor
llm = OpenAI(temperature=0)
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )

# compressed_docs = compression_retriever.get_relevant_documents(
#     "What did the president say about Ketanji Jackson Brown"
# )
# pretty_print_docs(compressed_docs)

# More built-in compressors: filters
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

# EmbeddingsFilter
# embeddings = OpenAIEmbeddings()
# embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=embeddings_filter, base_retriever=retriever
# )

# compressed_docs = compression_retriever.get_relevant_documents(
#     "What did the president say about Ketanji Jackson Brown"
# )
# pretty_print_docs(compressed_docs)

# Stringing compressors and document transformers together
# splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
# redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
# relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
# pipeline_compressor = DocumentCompressorPipeline(
#     transformers=[splitter, redundant_filter, relevant_filter]
# )
