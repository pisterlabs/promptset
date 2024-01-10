from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools.general_tools import QueryResourceTool, CreateQueryTool

def define_agent(tools):
    tools = [QueryResourceTool(), CreateQueryTool()]

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    conversational_memory = ConversationBufferWindowMemory(memory_key="chat_history", k=1, return_messages=True)

    query_agent = initialize_agent(agent='chat-conversational-react-description', 
                                  tools=tools, 
                                  llm=llm, 
                                  verbose=True, 
                                  max_iterations=3, 
                                  early_stopping_method='generate', 
                                  memory=conversational_memory)
    
    return query_agent