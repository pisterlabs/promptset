import pandas as pd
import os
from fastapi import  HTTPException, FastAPI
import os 
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd 
from loguru import logger
import matplotlib.pyplot as plt 
import seaborn as sns 
from fastapi.responses import FileResponse
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI 
from sqlalchemy import create_engine, MetaData
from langchain.llms import GooglePalm
from langchain.callbacks import get_openai_callback
from sqlalchemy import create_engine, MetaData
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine, MetaData
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from fastapi.responses import HTMLResponse

def count_tokens(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

# Load Customers.csv
Customers = pd.read_csv('Customers.csv')

# Load Customers_Employees.csv
Customers_Employees = pd.read_csv('Customers_Employees.csv')

# Load Product_Holding.csv
Product_Holding = pd.read_csv('Product_Holding.csv')

# Load RM_KRAs.csv
RM_KRAs = pd.read_csv('RM_KRAs.csv')

# Load contacthistory.csv
contacthistory = pd.read_csv('contacthistory.csv')

# Load Persona.csv
Persona = pd.read_csv('Persona.csv')

# Load Employees.csv
Employees = pd.read_csv('Employees.csv')

def load_csv_to_dataframes():
    dataframes = {}
    csv_files = [
        'Customers.csv',
        'Customers_Employees.csv',
        'Product_Holding.csv',
        'RM_KRAs.csv',
        'contacthistory.csv',
        'Persona.csv',
        'Employees.csv'
    ]
    
    for i, csv_file in enumerate(csv_files, 1):
        if os.path.isfile(csv_file):
            dataframe_name = f'df{i}'
            dataframes[dataframe_name] = pd.read_csv(csv_file)
            print(f"Loaded '{csv_file}' as '{dataframe_name}' dataframe.")
        else:
            print(f"File '{csv_file}' does not exist.")
    
    return dataframes
dataframes = load_csv_to_dataframes()
df = [dataframes[i] for i,j in dataframes.items()]

llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=80000,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

agent_pandas = create_pandas_dataframe_agent(llm, 
              df, verbose=True, ) 

# Create engines for both databases
engine = create_engine("sqlite:///data_KRA.sqlite")

db = SQLDatabase(engine)
sql_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

agent_sql = create_sql_agent(

    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5,
    handle_parsing_errors=True
)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query_pandas/")
def run_query(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")
    
    result = agent_pandas.run(query["query"])
    return {"query": query["query"], "result": result['output']}



@app.post("/query_sql/")
def query_sql(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")

    result = count_tokens(agent_sql, query["query"])
    return {"query": query["query"], "result": result['output']}

# @app.get("/html/")
# def send_html():
#     return HTMLResponse(content=html_content, status_code=200)


#################### Objective - 3 ##########################


#################### Objective - 1 ##########################

conversation_buf_report = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# Initialize conversation_buf with the initial message
initial_message_report = f'''Act like expert Data Analyst tell 
wheather the query is related to theoritical or graphical question
Output Should a string of theoritical or graphical'''

check_report = conversation_buf_report(initial_message_report)

@app.post("/query_report/")
def query_report(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")

    result = count_tokens(conversation_buf_report, query["query"])
    logger.info(result['response'])

    # Run the agent over multiple dataframe
    agent = create_pandas_dataframe_agent(llm , [Customers, Customers_Employees, Product_Holding , 
                                                RM_KRAs , contacthistory , Persona , Employees], verbose=True,
                                        return_intermediate_steps = True, max_iterations=5
                                                )
    ListOfCharts = ['bar', 'line', 'histogram' , 'pie' , 'scatter' , 'boxplot' , 'violinplot']
    for plot in ListOfCharts:
        try:
            answer = agent(f"make {plot} chart using seaborn and save the graph as 'sample.png' , Query:{query['query']}, Please do go through all the tables for the analysis")
            return answer
            return FileResponse('sample.png')
        except Exception as e:
            continue



@app.post("/query_for_table/")
def query_report(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")

    # Run the agent over multiple dataframe
    agent = create_pandas_dataframe_agent(llm , [Customers, Customers_Employees, Product_Holding , 
                                                RM_KRAs , contacthistory , Persona , Employees], verbose=True,
                                                max_iterations=7)

    answer = agent.run(query['query'])
    return answer 
        





