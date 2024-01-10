from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader  = WebBaseLoader("https://en.wikipedia.org/wiki/Leonardo_da_Vinci")

pages = loader.load_and_split()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(pages, embeddings)

retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever)

query = "What did leonardo da vinci study?"

result = qa_chain.run(query)

print(result)
