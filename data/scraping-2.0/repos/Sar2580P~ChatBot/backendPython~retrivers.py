import os, sys
sys.path.append(os.getcwd())
from langchain.retrievers import ContextualCompressionRetriever
from backendPython.utils import *
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains import RetrievalQA

db_path = 'backendPython/info_db'
db = Chroma(embedding_function = Embedding(), persist_directory= db_path)


_filter = LLMChainFilter.from_llm(llm)

retriever = db.as_retriever(search_kwargs={"k": 4, 'fetch_k': 20}, return_source_documents=True)
compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)

# qa_chain = RetrievalQA.from_chain_type(llm = llm , retriever=compression_retriever, chain_type="stuff")