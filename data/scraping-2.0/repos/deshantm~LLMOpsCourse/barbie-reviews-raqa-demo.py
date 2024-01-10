from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path='barbie-reviews.csv',
    source_column='Review_Url'
)

data = loader.load()
#sanity check
print("length of data: ")
print(len(data))



from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # the character length of the chunk
    chunk_overlap = 100, # the character length of the overlap between chunks
    length_function = len, # the length function
)
documents = text_splitter.transform_documents(data)

#print(documents)
#print length of documents
print("length of documents: ")
print(len(documents))

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

store = LocalFileStore("./cache/")

core_embeddings_model = OpenAIEmbeddings()

embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model,
    store,
    namespace=core_embeddings_model.model
)

vector_store = FAISS.from_documents(documents, embedder)

query = "How is Will Ferrell in this movie?"
embedding_vector = core_embeddings_model.embed_query(query)
docs = vector_store.similarity_search_by_vector(embedding_vector, k = 4)

for page in docs:
  print(page.page_content)

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

retriever = vector_store.as_retriever()

from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

print(qa_with_sources_chain({"query" : "How was Will Ferrell in this movie?"}))

print(qa_with_sources_chain({"query" : "Do reviewers consider this movie Kenough?"})

  