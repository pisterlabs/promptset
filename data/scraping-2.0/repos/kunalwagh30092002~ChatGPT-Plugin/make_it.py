from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import pinecone
import pdf2image

directory = 'file'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents
documents = load_docs(directory)
#print(len(documents))
def split_docs(documents, chunk_size=1536, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
docs = split_docs(documents)

def load_and_split_docs(directory):
    documents = load_docs(directory)
    docs = split_docs(documents)
    return docs

# Call the function and store the result
docs = load_and_split_docs(directory)

#embeddings = OpenAIEmbeddings(openai_api_key="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")

#index_name = "chatgptplugin"

#index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

