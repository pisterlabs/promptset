from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import time

start_time = time.time()
embeddings_open = OllamaEmbeddings(model="mistral")

loader = DirectoryLoader('datasmall/')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-gbif-datasets",
    embedding=embeddings_open,
    persist_directory="./embeddings_chroma_db",
)

vectorstore.persist()
print("--- %s seconds ---" % (time.time() - start_time))