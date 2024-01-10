from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
import requests
import os
# Load the .env file
load_dotenv()

@tool
def request_bing(query : str) -> str:
    """
    Searches the internet for additional information if the llm can not provide an answer by itself.
    Specifically useful when you need to answer questions about current events or the current state of the world.
    """
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_KEY")}
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    snippets_list = [result['snippet'] for result in data['webPages']['value']]
    snippets = "\n".join(snippets_list)
    return snippets

llm=ChatOpenAI(model_name="gpt-3.5-turbo" ,temperature=0.2)

tools = [request_bing]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, handle_parsing_errors=True)

print("Input something:")
while True:
    query = input()
    if query == "exit":
        break
    agent_chain.run(input=query)