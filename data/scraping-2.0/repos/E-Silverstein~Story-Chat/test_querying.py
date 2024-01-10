import os 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

openai_api_key = os.environ['OPENAI_KEY']
db = Chroma(persist_directory='embeddingsbooks/Alice_In_Wonderland.txt', embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=.5, streaming=True), retriever=retriever)
query = "You will act as Alice from Alice in Wonderland. You will speak to me as if you where her. Question: Who are you and where are you from? Alice: "
print(qa.run(query))