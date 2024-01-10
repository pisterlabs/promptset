import streamlit as st
from pathlib import Path
# from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

st.set_page_config(page_title="dailyLogChat", page_icon="🦜")
st.title("🦜 dailyLogChat")

# # User inputs
# radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
# selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
# if radio_opt.index(selected_opt) == 1:
#     db_uri = st.sidebar.text_input(
#         label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
#     )
# else:
#     db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
#     db_uri = f"sqlite:////{db_filepath}"

host="localhost"
port="5432"
database="ReportDB"
username="postgres"
password="postgres"

db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

# openai_api_key = st.sidebar.text_input(
#     label="OpenAI API Key",
#     type="password",
# )

import os
openai_api_key = open('key.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key

from dotenv import load_dotenv
load_dotenv()

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup agent
# llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

SQL query format example:
Question: "Who are the top 5 retailers for the month of May in terms of total play time?"
Query: SELECT "Retail Name", SUM("Total Play time") as total_play_time 
       FROM "dailyLog" 
       WHERE EXTRACT(MONTH FROM "Date") = 5 AND total_play_time IS NOT NULL
       GROUP BY "Retail Name" 
       ORDER BY total_play_time DESC 
       LIMIT 5

Observation: When ordering by a column in descending order, the top values will be the largest values in the column.
       
"""

SQL_FUNCTIONS_SUFFIX = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

# agent = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
    suffix=SQL_FUNCTIONS_SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    # agent_executor_kwargs = {'return_intermediate_steps': True}
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "You are connected with dailyLogDB. Ask questions!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)