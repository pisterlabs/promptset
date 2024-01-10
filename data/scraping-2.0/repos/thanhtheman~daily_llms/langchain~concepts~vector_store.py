
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import HNLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

loader = HNLoader("https://news.ycombinator.com/item?id=36645575")
data = loader.load()

vector_store = FAISS.from_documents(data, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()

response = retriever.get_relevant_documents("Why people hate langchain?")
print(len(response))