import os
os.environ["OPENAI_API_TYPE"]="azure"
os.environ["OPENAI_API_VERSION"]="2023-07-01-preview"
os.environ["OPENAI_API_BASE"]="https://oai-ai-r-force.openai.azure.com" # Your Azure OpenAI resource endpoint
os.environ["OPENAI_API_KEY"]="066287da630d473d91b7920aa341d418" # Your Azure OpenAI resource key
os.environ["OPENAI_CHAT_MODEL"]="GPT4-32k" # Use name of deployment

os.environ["SQL_SERVER"]="pccvsql-use2-dsoi-dev-fnd0001" # Your az SQL server name
os.environ["SQL_DB"]="fnd-configuration"
os.environ["SQL_USERNAME"]="sqlsa" # SQL server username 
os.environ["SQL_PWD"]="Crosby87Crosby87" # SQL server password 


from sqlalchemy import create_engine

driver = '{ODBC Driver 17 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';Server=tcp:' + os.getenv("SQL_SERVER")+'.database.windows.net;PORT=1433' + \
                ';DATABASE=' + os.getenv("SQL_DB") + \
                ';Uid=' + os.getenv("SQL_USERNAME")+ \
                ';Pwd=' + os.getenv("SQL_PWD") + \
                ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

db_engine = create_engine(odbc_str)

from langchain.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"),
                      deployment_name=os.getenv("OPENAI_CHAT_MODEL"),
                      temperature=0)

from langchain.prompts.chat import ChatPromptTemplate

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
          You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about Organization.
          Use following context to create the SQL query. Context:
          use config schema
         OrgOnboarding table contains information about onboarding per client.
         OrgOnboardingstatus table contains status about organization per onboarding
       
         If the question is about organizations per client, then left join ClientConfiguration, OrgOnboarding ,OrgOnboardingstatus and Organization tables.
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)

from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

db = SQLDatabase(db_engine,"config")

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

sqldb_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

sqldb_agent.run(final_prompt.format(
        question="count of org?"
  ))