from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import openai
import os


directory = 'content'

def load_docs(directory):
#   loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

documents = load_docs(directory)
docs = split_docs(documents)
# print(len(docs))

deployment_name = "codecompass-gpt"
emb_name = "codecompass_emb"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

# query = "Vector stores serve as a prevalent method for handling and searching?"
# matching_docs = db.similarity_search(query)

# print(matching_docs[0])

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()

os.environ["OPENAI_API_KEY"] = "f9e438c0871042d1bfdfb01cfd30d79e"
os.environ["OPENAI_API_BASE"] = "https://codecompassopenai.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = '2023-05-15'
model_name = "gpt-3.5-turbo"
llm = AzureChatOpenAI(model_name=model_name, deployment_name=deployment_name)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "What should a dog eat?"
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
print(retrieval_chain.run(query))

