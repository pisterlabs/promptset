### Importing Libraries
from google.cloud import bigquery
import streamlit as st
from sqlalchemy  import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import os
from langchain.agents  import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms.openai import OpenAI 
from langchain.agents import AgentExecutor 
from google.oauth2 import service_account

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
service_account_file = credentials

##### Code


# Change to where your service account key file is located
project  = "avian-light-210704"
dataset = "Pramod_Attribution"
table = "ROR"
sqlalchemy_url  = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'

client = bigquery.Client()
sql = """
SELECT * FROM `avian-light-210704.Pramod_Attribution.ROR`
LIMIT 1
"""
df = client.query(sql).to_dataframe()
print(df)
st.text(df)

# OPENAI_API_KEY = os.environ.get('SECRET_KEY')
# os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY ### Open API Key

# db = SQLDatabase.from_uri(sqlalchemy_url)
# #llm =OpenAI(temperature=0,model="text-embedding-ada-002	")
# llm =OpenAI(temperature=0,model="text-davinci-003") ### Type of embedding
# toolkit =SQLDatabaseToolkit(db=db,llm=llm)
# agent_executor =create_sql_agent(llm=llm,toolkit=toolkit,verbose=False,top_k=1000)

st.title("Querying RoR Dataset")

# agent_executor.run("""In ROR Dataset, can you check how many insititutions are active in Australia 
# after 2010. Can you also let me know how many number of institition in each type?
# """)
