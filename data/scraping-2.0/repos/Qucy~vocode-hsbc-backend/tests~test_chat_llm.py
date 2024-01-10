import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool

# prompt prefix append before the chat conversation
PROMPT_PREFIX = """
You are a customer service AI for HSBC Hongkong, your primary focus would be to assist customers with questions and issues related to HSBC Hongkong products and services. 
If a customer asks a question that is not related to HSBC Hongkong, politely inform them that I am only able to assist with HSBC Hongkong related questions.

You have access to the following tools:
"""

# tools
@tool("hsbc knowledge search tool")
def hsbc_knowledge_tool(input: str) -> str:
    """useful for when you need to answer questions about hsbc related knowledge"""
    # TODO need to be replace by vector search later
    return """
        Need to open a bank account with HSBC HK? You can apply with us if you:
            - are at least 18 years old
            - meet additional criteria depending on where you live
            - have proof of ID, proof of address
        If customer wants to apply online via mobile app:
            - They need to download the HSBC HK App to open an account online.
            - holding an eligible Hong Kong ID or an overseas passport
            - new to HSBC
    """


@tool("reject tool", return_direct=True)
def reject_tool(input: str) -> str:
    # LLM agent sometimes will not reject question not related to HSBC, hence adding this tools to stop the thought/action process
    """useful for when you need to answer questions not related to HSBC"""
    return """
    I'm sorry, but as a customer service chatbot for HSBC Hongkong, I am only able to assist with questions related to HSBC Hongkong products and services. 
    Is there anything else related to HSBC Hongkong that I can help you with?
    """

# load environment variables
load_dotenv()

# set up azure openai api
os.environ['OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["OPENAI_API_BASE"] = os.getenv('AZURE_OPENAI_API_BASE')
os.environ["OPENAI_API_TYPE"] = os.getenv('AZURE_OPENAI_API_TYPE')
os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_API_VERSION')

# create llm model
llm = AzureChatOpenAI(deployment_name=os.getenv('AZURE_OPENAI_API_ENGINE'), model=os.getenv('AZURE_OPENAI_API_MODEL'))

# create tools
tools = [hsbc_knowledge_tool, reject_tool]

# create memory, window size = 10
memory = ConversationBufferWindowMemory(memory_key='chat_history',k=10, return_messages=True)

# create agent
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "system_message": PROMPT_PREFIX
    }
)

print(agent_chain.agent.llm_chain)

agent_chain.run(input="How can I open an account at HSBC HK ?")

agent_chain.run(input="How can I open an account at CITI BANK ?")