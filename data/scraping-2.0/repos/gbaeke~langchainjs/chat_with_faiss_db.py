from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# load env vars
load_dotenv()

# OpenAI embeddings
embeddings = OpenAIEmbeddings(client=None)

# load FAISS db
db = FAISS.load_local("faiss_db", embeddings)

# memory for chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# llm
llm = ChatOpenAI(temperature=0, client=None)

# init retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)

while True:
    question = input("\033[31mQuestion (type 'exit' to quit):\033[0m ")
    if question == "exit":
        break
    result = qa({"question": question})
    print("\033[32m" + result["answer"] + "\033[0m")
    