from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

database = Chroma(persist_directory="./data", embedding_function=embeddings)

retriever = database.as_retriever()

qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

result = qa("飛行車の最高速度は?")
print(result["result"])
print(result["source_documents"])
