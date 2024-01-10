from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import dotenv
# from langchain.agents import Tool,tool


dotenv.load_dotenv('apartment_price_system_update/.env')


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
embeddings = OpenAIEmbeddings()
db=FAISS.load_local('apartment_price_system_update/docs/aps.db',embeddings)
# db=FAISS.load_local('docs/aps.db',embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    model,
    db.as_retriever(searchtype="similarity_search", k=5),
    memory=memory)

# @tool("aps_query",return_direct=True)
def aps(query):
    """usefull for know apartment price from address"""
    global qa
    result = qa.run({"question": query})   

    print(result) 
    return result
