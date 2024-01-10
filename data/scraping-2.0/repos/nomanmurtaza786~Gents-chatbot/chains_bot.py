import os

from dotenv import load_dotenv
from langchain.agents import (AgentExecutor, AgentType, Tool, create_sql_agent,
                              initialize_agent, load_tools)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import APIChain, ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase

from supabase_db import get_vector_store_retriever

load_dotenv()

openApiKey: str = os.getenv("OPENAI_API_KEY", "default_key")
huggingFaceApikey = os.getenv("API_KEY", "default_key")
openAillm = OpenAI(
    model="text-davinci-003",
    openai_api_key=openApiKey,
)
template = """ 
    Question: {question} 
    think step by step
    Answer: 
    """
prompt = PromptTemplate(template=template, input_variables=["question"])

chatLLM = ChatOpenAI(temperature=0.1,)
llm_chain = LLMChain(llm=chatLLM, prompt=prompt,verbose=True,)
#print("predict", chatLLM.predict('Captial of USA'))

crc = ConversationalRetrievalChain.from_llm(llm=chatLLM, retriever=get_vector_store_retriever(), verbose=True, )

api_chain = APIChain.from_llm_and_api_docs(llm=chatLLM, api_docs='' ,verbose=True,)


def get_answer(question: str, chat_history: list = []):
    result = crc({"question": question, "chat_history": chat_history})
    return result

def callingApiChain(question: str, chat_history: list):
    result = crc({"question": question, "chat_history": chat_history})
    return result["answer"]

## database connect string from env file
db_connect_string = os.getenv("DB_Connection_Str", "default_key")
db = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost:5432/HR")
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

def get_answer_from_agent(question: str, chat_history: list = []):
    result = agent_executor(question)
    return result

tools = [
    Tool(
        name="sql_agent",
        description='use to employee data from database such as active employees, performance rating, location, department, etc.',
        func=get_answer_from_agent,
        
    ),
      Tool(
        name="resume_reader",
        description='use to read resume and extract information such as name, email, phone, skills, etc.',
        func=get_answer,
    )
]

agents = initialize_agent(tools=tools, verbose=True,  llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True), )


def run_multiple_agents(question: str, chat_history: list = []):
    result = agents.run(question)
    return result
# print("predict", get_answer_from_agent('Tell me best location with respect to employees performance. IF rating A is consider as best performer'))