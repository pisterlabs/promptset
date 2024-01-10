from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
import os
import openai

# llm = AzureOpenAI(
#     openai_api_base=os.getenv("OPENAI_API_BASE"),
#     openai_api_version="version",
#     deployment_name="deployment name",
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     openai_api_type="azure",
# )

directory = '/home/sam/Insync/sam.garfield@gamp.ai/Google Drive - Shared drives/TNano'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vector_store = FAISS.from_documents(docs, embeddings)

#save the vectorstore to disk


# chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     return_source_documents=True
# )
# while True:
#     query = input("Input your question\n")
#     result = chain(query)
#     print("Answer:\n")
#     print(result['answer'])