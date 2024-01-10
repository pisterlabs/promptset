import dotenv

dotenv.load_dotenv()

from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

# Text Splitting & Docloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

loaders = [
    TextLoader('langchain_blog_posts/blog.langchain.dev_announcing-langsmith_.txt'),
    TextLoader('langchain_blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt'),
]
docs = []
for l in loaders:
    docs.extend(l.load())

print(f'Loaded {len(docs)} documents')

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=embeddings
)

# The storage layer for the parent documents
store = InMemoryStore()

full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

full_doc_retriever.add_documents(docs, ids=None)

retrieved_docs = full_doc_retriever.get_relevant_documents("what is langsmith")

print(retrieved_docs[0].page_content)

# Retrieving larger chunks

# This text splitter is used to create the parent documents - The big chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents - The small chunks
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings)

# The storage layer for the parent documents
store = InMemoryStore()

big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

big_chunks_retriever.add_documents(docs)

print(len(list(store.yield_keys())))

sub_docs = vectorstore.similarity_search("what is langsmith")

print(sub_docs[0].page_content)

retrieved_docs = big_chunks_retriever.get_relevant_documents("what is langsmith")

print(retrieved_docs[0].page_content)

print(retrieved_docs[1].page_content)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(),
                                 chain_type="stuff",
                                 retriever=big_chunks_retriever)

query = "What is Langsmith?"
response = qa.run(query)
print(response)