from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


loader = WebBaseLoader('https://en.wikipedia.org/wiki/2023_Hawaii_wildfires')
data = loader.load()

ollama = Ollama(base_url="http://localhost:11434",model="llama2")
#ollama = Ollama(model="samantha-mistral")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorestore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorestore.as_retriever())

question = "What did the Natinal Interagency Fire Center forecast?"
print(qachain({"query": question})['result'])
## That shit works 🚀🚀🚀!!
