from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import chroma
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatGooglePalm
from langchain.chains import RetrievalQA



directory = '/Users/akhilkumarp/development/personal/github/LLM_Journey/RagOverCode'

# loader = GenericLoader.from_filesystem(
#     directory,
#     glob="*",
#     suffixes=[".pdf"],
# )
# documents = loader.load()
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = chroma.from_documents(docs, embeddings)

query = "What are the different types of concepts explained"
matching_docs = db.similarity_search(query)

matching_docs[0]

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()
import os
os.environ["OPENAI_API_KEY"] = "key"

# model_name = "gpt-3.5-turbo"
# llm = ChatOpenAI(model_name=model_name)
model_name = "text-bison-001"
llm = ChatGooglePalm(model_name=model_name)
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "What are the emotional benefits of owning a pet?"
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)
answer
retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
retrieval_chain.run(query)


