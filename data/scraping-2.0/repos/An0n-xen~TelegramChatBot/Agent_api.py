import os
from dotenv import load_dotenv, find_dotenv

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseSettings
from pydantic import BaseModel

from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

# loading env files
load_dotenv(find_dotenv(), override=True)

# Initializing FastAPI
app = FastAPI()


class Settings(BaseSettings):
    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


# Setting type sendMessage
class sendMessage(BaseModel):
    update_id: int
    message: str


# loading Gmail toolkit
toolkit = GmailToolkit()
tools_gmail = toolkit.get_tools()

# loading search engine
search = SerpAPIWrapper(serpapi_api_key=os.environ.get("SERP_API_KEY"))


def Agent_gmail():
    # Initializing gmail agent
    llm_gmail = OpenAI(temperature=0)
    agent_gmail = initialize_agent(
        tools=tools_gmail,
        llm=llm_gmail,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    return agent_gmail


# Initializing gmail agent
agent_gmail = Agent_gmail()


def Agent_main():
    # Setting up new template for agent
    system_message = """
    Assistant is a large language model designed for Suprcede.

    Assistant's name is Afia, you are a Business Development Representative from Boxpark Real estate , a real estate company

    Assistant is designed to be able to assist with a wide range of tasks but geared toward real estate, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics but mostly real estate. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Assistant is always trying to direct conversation towards it purpose, as a Development representative of Boxpaark Real estate

    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    """

    # Setting tools for main agent
    tools_main = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="Gmail",
            func=agent_gmail.run,
            description="Useful when need to send an email to someone, this tool is an agent to just input the prompt as you recieved it",
        ),
    ]

    # Initializing main agent
    llm_main = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY_SUPRCEDE"),
        temperature=0.3,
        model="gpt-4",
    )

    conversational_memory = ConversationBufferMemory(
        memory_key="chat_history", k=10, return_messages=True
    )

    agent_main = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools_main,
        llm=llm_main,
        memory=conversational_memory,
        max_iterations=3,
    )

    new_prompt = agent_main.agent.create_prompt(
        tools=tools_main, system_message=system_message
    )

    agent_main.agent.llm_chain.prompt = new_prompt

    return agent_main


# Initializing agent_main
agent_main = Agent_main()


def botResponse(msg: str) -> str:
    response = agent_main.run(msg)
    return response


@app.get("/")
def root():
    return {"Hello this Supersede agent home"}


@app.post("/")
def chatbot(in_message: sendMessage):
    message = in_message.message
    bot_response = botResponse(message)
    return {"message": bot_response}
