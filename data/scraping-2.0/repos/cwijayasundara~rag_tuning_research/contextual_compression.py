import dotenv

dotenv.load_dotenv()

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain.embeddings import OpenAIEmbeddings

loaders = [
    TextLoader('langchain_blog_posts/blog.langchain.dev_announcing-langsmith_.txt'),
    TextLoader('langchain_blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt'),
]
docs = []
for l in loaders:
    docs.extend(l.load())

print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()


# Helper function for printing docs

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# plain retriever
retriever = FAISS.from_documents(texts, embedding).as_retriever()

docs = retriever.get_relevant_documents("What is LangSmith?")
pretty_print_docs(docs)

# Adding contextual compression with an LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# making the compressor
llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# it needs a base retriever (we're using FAISS Retriever) and a compressor (Made above)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What is LangSmith?")
pretty_print_docs(compressed_docs)

# EmbeddingsFilter

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(embeddings=embeddings,
                                     similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                       base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What is LangSmith")
pretty_print_docs(compressed_docs)

# Pipelines
# Stringing compressors and document transformers together

from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# making the pipeline
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                       base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What is LangSmith")
pretty_print_docs(compressed_docs)
