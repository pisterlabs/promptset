import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

"""
컨텍스트 압축은 문서 검색 시스템이 특정 질문에 맞게 정보를 정제하는 것을 말합니다.
전체 문서를 반환하는 대신 쿼리와 관련된 내용만을 압축해서 보여줍니다.
베이스 리트리버와 문서 압축기가 필요한데, 이 과정에서 베이스 리트리버가 질문을 처리하고 초기 문서를 가져오고,
문서 압축기가 해당 문서들을 요약하거나 필요 없는 것들을 제거하여 리스트를 줄입니다.
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter



documents = TextLoader('../dataset/state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)[:10]
retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai_api_key)).as_retriever()

llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
print(compressed_docs)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
print(compressed_docs)