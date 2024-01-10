from dotenv import load_dotenv
import logging
import sys
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index import GPTVectorStoreIndex
from llama_index import download_loader
from llama_index import StorageContext, load_index_from_storage
from llama_index.chat_engine.types import ChatMode

def load_bot():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, system_prompt="You are a DYI expert in a large hardware retail chain, you are suppose to answer customer query based on your knowledge and product information avaiable for you.")
    # bot = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, system_prompt="You are a DYI expert in a large hardware retail chain, you are suppose to answer customer query based on your knowledge and product information avaiable for you.")
    # tools = [
    #     Tool(
    #         name="LlamaIndex",
    #         func=lambda q: str(query_engine.query(q)),
    #         description="useful when you want to answer questions about product information such as price, link, description, etc. The input to this tool should be a complete english sentence.",
    #         return_direct=True,
    #     ),
    # ]
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # llm = ChatOpenAI(temperature=0.5)
    # agent_executor = initialize_agent(
    #     tools, llm, agent="conversational-react-description", memory=memory, tags=["LlamaIndex"]
    # )
    # return agent_executor
    return query_engine

def create_index():
    SimpleCSVReader = download_loader("SimpleCSVReader")
    loader = SimpleCSVReader()
    documents = loader.load_data(file=Path('./data'))
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

if __name__ == "__main__":
    bot = load_bot()
    # response = bot.run(input="I would like to clean up my deck what product should I buy?")
    # print(response)
    # response = bot.run(input="which one is the cheapest?")
    # print(response)
    response = bot.query("I would like to clean up my deck what product should I buy?") 
    print(response.response)
    response = bot.query("which one is the cheapest?") 
    print(response.response)
