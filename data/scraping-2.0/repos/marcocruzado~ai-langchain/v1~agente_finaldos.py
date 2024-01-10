from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain 
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
import os

# Otra forma desde una URL
from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.environ["OPENAI_API_KEY"])

#### FUENTES DE INFORMACION"#####################

# tiene que buscar en google y devolver el primer resultado todas las busquedas sera en Peru y en español
search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"], params={
                        "engine": "google", "google_domain": "google.com", "gl": "pe", "hl": "es-419"})

# -------------------------------------------------

db = SQLDatabase.from_uri('postgresql+psycopg2://postgres:root@localhost:5432/postgres',
                                include_tables=['promociones'])

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
)

# -------------------------------------------------

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Search in Database",
        func=agent_executor.run,
        description="cuando necesites saber sobre promociones de los establecimientos que el club el comercio,numeros de telefono, correos, etc",
    ),
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm=OpenAI(temperature=0),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    handle_parsing_errors=True,
)


while True:
    user_input = str(input("User:"))
    agent_output = agent.run(input=str(user_input))
    print("Agent:", agent_output)