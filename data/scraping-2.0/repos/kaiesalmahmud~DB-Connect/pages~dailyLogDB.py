import os
import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import streamlit as st

API_KEY = open('key.txt', 'r').read().strip()
DB_PASSWORD = open('pass.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

openai.api_key = API_KEY

from dotenv import load_dotenv
load_dotenv()


# Implement: If null value is found at the top while trying to sort in descending order, try to look for the next non-null value.

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


def get_response(input_text):
    response = agent_executor(input_text)

    # print(response['intermediate_steps'][1][0].tool)
    # print(response['intermediate_steps'][-1][0].tool)
    # print(response['output'])


    if response['intermediate_steps'][1][0].tool == 'sql_db_schema':
        schema = response['intermediate_steps'][1][1]
    else: schema = None

    if response['intermediate_steps'][-1][0].tool == 'sql_db_query':
        query = response['intermediate_steps'][-1][0].tool_input
        query_output = response['intermediate_steps'][-1][1]
    else: query, query_output = None, None

    answer = response['output']

    return schema, query, query_output, answer

def explain(query, schema, query_output):

    message_history = [{"role": "user", "content": f"""You are a SQL query explainer bot. That means you will explain the logic of a SQL query. 
                    There is a postgreSQL database table with the following table:

                    {schema}                   
                    
                    A SQL query is executed on the table and it returns the following result:

                    {query_output}

                    I will give you the SQL query executed to get the result and you will explain the logic executed in the query.
                    Make the explanation brief and simple. It will be used as the explanation of the results. Do not mention the query itself.
                    No need to explain the total query. Just explain the logic of the query.
                    Reply only with the explaination to further input. If you understand, say OK."""},
                   {"role": "assistant", "content": f"OK"}]

    message_history.append({"role": "user", "content": query})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    explaination = completion.choices[0].message.content
    return explaination

host="localhost"
port="5432"
database="ReportDB"
username="postgres"
password="postgres"

# # Create the sidebar for DB connection parameters
# st.sidebar.header("Connect Your Database")
# host = st.sidebar.text_input("Host", value=host)
# port = st.sidebar.text_input("Port", value=port)
# username = st.sidebar.text_input("Username", value=username)
# password = st.sidebar.text_input("Password", value=password)
# database = st.sidebar.text_input("Database", value=database)
# # submit_button = st.sidebar.checkbox("Connect")

db = SQLDatabase.from_uri(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
    suffix=SQL_FUNCTIONS_SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    agent_executor_kwargs = {'return_intermediate_steps': True}
)

# Create the main panel
st.title("connectDB :star2:")
st.subheader("You are connected to AD-IQ DailyLog database!!")
st.caption("The database contains the Daily Log data for AD-IQ screens from April 1st to June 17th")

with st.expander("Database properties"):
    st.text("""
            "Date"
            "Retail Name"
            "Zone"
            "Location"
            "Total Play time"
            "Efficiency (percent)"
            "Start Time"
            "End Time"
            "Device Offline time"
            "Remarks"
            """)

with st.expander("FAQs"):
    st.text("""
    1. Describe the database.
    2. What is the timeline of the data present?
    3. What is the average total play time for the month of April?
    4. Who are the top 5 retailers for the month of May in terms of total play time?
    5. How many areas are the shops located at?
    6. What is the combined total play time for 5th May?
    7. List the top 5 areas with least average efficiency.
    8. List of most not opened shops for the month of April.
    9. Which shops has most count of playtime of less than 10 hours?
    10. Which shops has the most count of start time after 10 am?
    """)

# Get the user's natural question input
question = st.text_input(":blue[Ask a question:]", placeholder="Enter your question.")

# Create a submit button for executing the query
query_button = st.button("Submit")

# Execute the query when the submit button is clicked
if query_button:

    # Display the results as a dataframe
    # Execute the query and get the results as a dataframe
    try:
        with st.spinner('Calculating...'):
            print("\nQuestion: " + str(question))
            # print(str(question))
            schema, query, query_output, answer = get_response(question)

            if query:
                explaination = explain(query, schema, query_output)
            else: explaination = None

            # explaination = explain(query, schema, query_output)

        # if query:
        #     print("\nExplaination: " + str(explaination))

        print("\nExplaination: " + str(explaination))

        st.subheader("Answer :robot_face:")
        st.write(answer)

        try:
            if query:

                st.divider()
                # st.caption("Query:")
                # st.caption(query)

                st.caption("Explaination:")
                st.caption(explaination)

                st.divider()
        except Exception as e:
            print(e)

        st.info(":coffee: _Did that answer your question? If not, try to be more specific._")
    except Exception as e:
        print(e)
        st.warning(":wave: Please enter a valid question. Try to be as specific as possible.")
