from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import time as timer

pdf_path = "/Users/mghifary/Work/Code/AI/data/cerita-rakyat-nusantara2.pdf"
loader = PyMuPDFLoader(pdf_path)
data = loader.load()

llm = Ollama(
    base_url='http://localhost:11434',
    model='mistral',
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Define vectostore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Define RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm, 
    retriever=vectorstore.as_retriever(),
    verbose=True
)

# Define prompt
query = "Explain about Prabu Siliwangi, an historical character from West Java"

print(f"Query: {query}")
# docs = vectorstore.similarity_search(query)
# print(f"Docs (similarity search results): {docs}")

# Run the chain
start_t = timer.time()
response = chain({"query": query})
elapsed_t = timer.time() - start_t
print(f"\n\nElapsed time: {elapsed_t}")





