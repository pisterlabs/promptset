#an API to call the text2sql model
import os
from langchain.chat_models import ChatOllama
from langchain.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms.openai import OpenAI

SQL_FAIL_MESSAGE = "SQL_ERROR"
def read_api_key(file_path):
    '''read the api key from the file
    :param file_path: the path of the file
    '''
    with open(file_path, 'r') as file:
        return file.read().strip()

REPLICATE_API_TOKE = read_api_key('../API_Key/REPLICATE_API_TOKEN.txt')
OPENAI_API_KEY = read_api_key('../API_Key/OPENAI_API_KEY.txt')

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def model_select(model_name):
    if model_name == "llama2_chat":
        return ChatOllama(model="llama2:13b-chat")
    elif model_name == "llama2_code":
        return ChatOllama(model="codellama:7b-instruct")
    elif model_name == "gpt4":
        return ChatOpenAI(model='gpt-4-0613')
    elif model_name == "gpt3":
        return ChatOpenAI(model='gpt-3.5-turbo-1106')
    else:
        return ChatOpenAI(model='gpt-3.5-turbo-1106')

def init(model_name,db_name):
    model = model_select(model_name)

    # Replicate API
    replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llama2_chat_replicate = Replicate(
    model=replicate_id, model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )
    # Database

    db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    return model,db

def text2sql(model_name,db_name,question):
    model,db = init(model_name,db_name)
    # Using Closure desgin pattern to pass the db to the model
    def get_schema(_):
        return db.get_table_info()
    template = """Based on the table schema below, write a SQLite query that would answer the user's question:
    {schema}
    
    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return sql_response.invoke({"question": question})  

def execute_sql(query,db_name):
    db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    update_action_list = ['UPDATE','ADD','DELETE','DROP','MODIFY','INSERT']
    try:
        if any(item in query for item in update_action_list)==False:# no update actions
            result = db.run(query)
            if result:
                return result
            else:
                return "No results found."
        else: return 'Finished' #update actions return no result but "Finished"
    except Exception as e:
        error_message = str(e)
        print(SQL_FAIL_MESSAGE,error_message)
        return SQL_FAIL_MESSAGE

def sqlresult2text(model_name,db_name,question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model,db = init(model_name,db_name)
    def get_schema(_):
        return db.get_table_info()
    ## To natural language
    
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response.invoke({"question": question,"query":sql_query,"response":sql_result})


def text2sql_end2end(model_name,db_name,question):
    model,db = init(model_name,db_name)
    # Prompts
    # 
    def get_schema(_):
        return db.get_table_info()

    def run_query(query):
        print("running query\n", query)
        try:
            result = db.run(query)
            if result:
                print("successfully run query")
                return result
            else:
                return "No results found."
        except Exception as e:
            error_message = str(e)
            print(SQL_FAIL_MESSAGE)
            return SQL_FAIL_MESSAGE

    template = """Based on the table schema below, write a SQLite query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    ## To natural language
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    full_chain = (
        RunnablePassthrough.assign(query=sql_response).assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | model
    )

    # execute the model 
    return full_chain.invoke({"question": question})

def sql_agent(question):
    db = SQLDatabase.from_uri("sqlite:///./Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model='gpt-3.5-turbo-1106',temperature=0))
    agent_executor = create_sql_agent(
    llm=ChatOpenAI(model='gpt-3.5-turbo-1106',temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
    agent_executor.run(
    question
)


def sql_explaination(model_name,db_name,question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model,db = init(model_name,db_name)
    def get_schema(_):
        return db.get_table_info()
    ## To natural language
    
    template = """Based on the table schema below, question, sql query, and sql response, explain the sql query step by step:
    {schema}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response.invoke({"question": question,"query":sql_query,"response":sql_result})
# sample to execute the model 
# question = "What are names of artists"
# question='Which country has the most actors'
# question='Find the most rented films'
question='Find Films That Were Rented More Than Once on the Same Day'
#example of using the model
# example 1: auto agent
# print(sql_agent(question))
# example 2: end 2 end
# print(text2sql_end2end("gpt3","Chinook",question))
# example 3: step by step
db_name = "Sakila_master"
sql= text2sql("gpt3",db_name,question)
print('sql:\n',sql)
result = execute_sql(sql,db_name)
print('result:\n',result)
text = sqlresult2text("gpt3",db_name,question,sql,result)
print(text)

explain=sql_explaination("llama2_chat",db_name,question,sql,result)
print(explain)