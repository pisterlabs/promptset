from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools, Tool

#chat model imports
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
#template imports
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#for the agent
from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase #,
    #SQLDatabaseChain,
)

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

import os

#environmental variable
API_KEY = "insert openAI key here"
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["SERPAPI_API_KEY"] = "insert openAI key here"

chat = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613")
llm = ChatOpenAI(openai_api_key = "insert openAI key here")
llm = ChatOpenAI(temperature = 0.9, model = "gpt-3.5-turbo-0613")

template = ("You are a personal nurse trying to identify dermatolgy issues and provide solutions to the patient before they go to seek medical care. You are made to provide help to those that cannot afford to seek treatment first.")
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.format_messages(text = print("Welcome to Derma Fix! \nWe're here to help you identify skin problems and provide appropiate treatment \nbefore seeking a trained professional. \n"))

chain = LLMChain(llm = chat, prompt = chat_prompt)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a dermatologist assistant that helps provide a diagnosis and solution to patients."
    ),
    MessagesPlaceholder(variable_name= "chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# load_tools(["serpapi", "llm-math"], llm=llm)
# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()
# db = SQLDatabase.from_uri("sqlite:///chinook.db")
#db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# tools = [
#     Tool(
#         name="Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events. You should ask targeted questions",
#     ),
#     Tool(
#         name="Calculator",
#         func=llm_math_chain.run,
#         description="useful for when you need to answer questions about math",
#     ),
#     Tool(
#         name="FooBar-DB",
#         func=db_chain.run,
#         description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
#     ),
#     Tool(
#         name = "Current Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events or the current state of the world"
#     ),
#]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#agent = initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
#agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# chain.run(
#     text = "text",
#     deadline = deadline,
#     company = company,
#     role = role,
#     difficulty=difficulty,
#     prompt=f"Provide either a interview question based on {role} for a leetcode prompt based on {difficulty} and {role}. You are an interviewing me who wants to work at {company} as a {difficulty} {role}. Do not include both at the same time.",
# )
#
agent.run(f"Provide a diagnosis and solution.")
agent_chain.run(input="answer")




