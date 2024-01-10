from langchain.retrievers import ParentDocumentRetriever

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from src.vectorize.process import load_chroma

loader = DirectoryLoader('../../static')
docs = loader.load()


# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = load_chroma()
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=load_chroma(),
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

list(store.yield_keys())

sub_docs = vectorstore.similarity_search("小栓母亲是谁")

print(sub_docs[0].page_content)