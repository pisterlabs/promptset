# Helper function for printing docs
from langchain.text_splitter import CharacterTextSplitter
from src.vectorize.process import local_embedding
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


loader = DirectoryLoader('../../static')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=16)
texts = text_splitter.split_documents(documents)
embedding = local_embedding()
retriever = FAISS.from_documents(texts, embedding).as_retriever()

docs = retriever.get_relevant_documents("小栓的父亲叫什么？")
pretty_print_docs(docs)

print("-----------use compression-----------")
llm = OpenAI(temperature=0, openai_api_key="sk-xxx")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("小栓的父亲叫什么？")
pretty_print_docs(compressed_docs)
