# Langchain
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import (AgentExecutor, ZeroShotAgent, load_tools)
from langchain.tools import Tool
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from snowflake.snowpark import Session
from sqlalchemy.dialects import registry


registry.load('snowflake')

# other imports
import logging
import os
import sys
from dotenv import load_dotenv, find_dotenv
# from snowflake.connector.errors import ProgrammingError 


# Streamlit
import streamlit as st


load_dotenv(find_dotenv(), override=True)

# Set up root logger to output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("This is an informational message.")

st.title("🏥 Catapult-Healthcare Bot")

#  openai_organization="YOUR_ORGANIZATION_ID"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACCOUNT = os.environ.get("ACCOUNT")
USER = os.environ.get("USER")
PASSWORD = os.environ.get("PASSWORD")
SCHEMA = os.environ.get("SCHEMA")
WAREHOUSE = os.environ.get("WAREHOUSE")
ROLE = os.environ.get("ROLE")
DATABASE = os.environ.get("DATABASE")


snowflake_url = f"snowflake://{USER}:{PASSWORD}@{ACCOUNT}/{DATABASE}/{SCHEMA}?warehouse={WAREHOUSE}&role={ROLE}"

db = SQLDatabase.from_uri(snowflake_url)
# sample_rows_in_table_info=1,
# , include_tables=['HEALTHRECORDDATA','HEALTHRECORDDATA_ATTRIBUTES']

# we can see what information is passed to the LLM regarding the database
print(db.table_info)
    
llm = ChatOpenAI(streaming=True,
                openai_api_key=OPENAI_API_KEY, 
                 model="gpt-4-1106-preview",
                 temperature=0)


DESCRIPTION = (
"You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.\n"
"The user will ask questions, for each question you should respond with an sql query based on the question and the table.\n"
"Do not use this tool to make charts\n"

"{context}"

"Here are 6 critical rules for the interaction you must abide:"
"<rules>"
"1. You MUST MUST wrap the generated sql code within ``` sql code markdown in this format e.g"
"```sql"
"(select 1) union (select 2)"
"```"
"2. If I don't tell you to find a limited set of results in the sql query or question, you MUST NOT limit the number of responses."
"3. Text / string where clauses must be fuzzy match e.g ilike %keyword%"
"4. Make sure to generate a single snowflake sql code, not multiple. "
"5. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names"
"6. DO NOT put numerical at the very front of sql variable."
"</rules>"
"Don't forget to use like %keyword% for fuzzy match queries (especially for variable_name column)"
"and wrap the generated sql code with ``` sql code markdown in this format e.g:"
"```sql"
"(select 1) union (select 2)"
"```"

"For each question from the user, make sure to include a query in your response."
)



QUALIFIED_TABLE_NAME = "CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA"

TABLE_DESCRIPTION = """
This table is an electronic health record (EHR) system or a patient health database which has clinical or healthcare data from patients.
"""

PREFIX = "Complete this task in the best way possible. To be able to do it, you have to use the following tools:"


METADATA_QUERY = "SELECT VARIABLE_NAME, DEFINITION FROM CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA_ATTRIBUTES;"

SUFFIX = """
"Begin! They are going to ask you for data and also ask you to create charts from that data.\n"
"Try to divide the input question into two parts: the part that refers to querying the database and the part of the input question where you are asked to graph the results of the question to the database."
"You will be acting as an AI Snowflake SQL Expert named Catapult Health Bot."
"Your goal is to give correct, executable sql query to users."
"You will be replying to users who will be confused if you don't respond in the character of Catapult Health Bot."
"You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag."

"{context}"

" You MUST MUST wrap the generated python code within ``` python code markdown in this format e.g"
"```python"
"import pandas as pd"
"df = pd.DataFrame([(2878, datetime.date(2023, 2, 22)), (2909, datetime.date(2023, 2, 23)), (2977, datetime.date(2023, 2, 24)), (4184, datetime.date(2023, 2, 21)), (25123, datetime.date(2023, 2, 20)), (31730, datetime.date(2023, 2, 19)), (24394, datetime.date(2023, 2, 18))], columns=['cantidad_lineas_diferentes', 'fecha'])"
"st.bar_chart(df, x='fecha', y='cantidad_lineas_diferentes')
"```
"Use the 'Python_REPL' tool to make the charts, whenever the input begins in one of the following possible ways: " 
"show a bar chart..., show a pie chart..., show a line chart... plot a bar chart..., plot a pie chart..., plot a line chart... Most of the time, in the responses, you will have to show charts. "
"Don't explain the query before retrieving it,just show the SQL Query, the dataframe result and if it's the case, the chart required for the user. You MUST NOT be verbose on the explanation, just show the results."


"Use the 'Python_REPL' tool to make the charts, whenever the input begins in one of the following possible ways: "
"show a bar chart..., show a pie chart..., show a line chart... plot a bar chart..., plot a pie chart..., plot a line chart... Most of the time, in the responses, you will have to show charts. "
"To make these charts, you first need to create a pandas dataframe with the result of the database query. Intelligently select the columns of this dataframe that will be used to display the chart. For example:"
"import streamlit as st"
"import pandas as pd"
"import datetime"
"import plotly.express as px"
"df = pd.DataFrame([(2878, datetime.date(2023, 2, 22)), (2909, datetime.date(2023, 2, 23)), (2977, datetime.date(2023, 2, 24)), (4184, datetime.date(2023, 2, 21)), (25123, datetime.date(2023, 2, 20)), (31730, datetime.date(2023, 2, 19)), (24394, datetime.date(2023, 2, 18))], columns=['cantidad_lineas_diferentes', 'fecha'])"
"st.bar_chart(df, x='fecha', y='cantidad_lineas_diferentes')"
"These charts should be made using as input the results of the query to the table <tableName>. "
"To make these charts, use the Plotly library, importing only the necessary libraries. "
"Do not import libraries that will not be used. Import one library at a time. "
"There should be only one import per line. "
"Use the 'Python_REPL' tool for the final response, after obtaining the response to the database query. "
"The final response of the 'Python_REPL' tool should not be in triple quotes, the python code should be executed. "
"When you do not use the 'Python_REPL' tool, respond to the question in English. Use the 'Python_REPL' tool when the input begins in the following way: "
"Show a line chart..."

"Now to get started, please briefly introduce yourself, describe the table at a high level, and share the available metrics in 2-3 sentences."
"Then provide 3 example questions using bullet points."
"""


@st.cache_data(show_spinner=False)
def get_table_context(table_name: str, table_description: str, metadata_query: str = None):
    table = table_name.split(".")

    # Your connection parameters
    connection_parameters = {
        "account": ACCOUNT,
        "user": USER,
        "password": PASSWORD,
        "role": ROLE,
        "warehouse": WAREHOUSE,
        "database": DATABASE,
        "schema": SCHEMA
    }

    # Create a Snowflake session
    session = Session.builder.configs(connection_parameters).create()

    # Execute the query
    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
    """
    result = session.sql(query).collect()

    # Process the result
    columns = "\n".join(
        [
            f"- **{row['COLUMN_NAME']}**: {row['DATA_TYPE']}"
            for row in result
        ]
    )

    context = f"""
    Here is the table name <tableName> {'.'.join(table)} </tableName>

    <tableDescription>{table_description}</tableDescription>

    Here are the columns of the {'.'.join(table)}

    <columns>\n\n{columns}\n\n</columns>
    """

    if metadata_query:
        metadata_result = session.sql(metadata_query).collect()
        metadata = "\n".join(
            [
                f"- **{row['VARIABLE_NAME']}**: {row['DEFINITION']}"
                for row in metadata_result
            ]
        )
        context += f"\n\nAvailable variables by VARIABLE_NAME:\n\n{metadata}"

    return context



def get_system_prompt():
    table_context = get_table_context(
        table_name=QUALIFIED_TABLE_NAME,
        table_description=TABLE_DESCRIPTION,
        metadata_query=METADATA_QUERY,
    )
    return SUFFIX.format(context=table_context)


db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,  # Show its work
    return_direct=True,  # Return the results without sending back to the LLM
)


sql_query_tool = Tool(name="text_to_sql",
                      func=db_chain.run,
                      description=DESCRIPTION)


python_repl_tool = PythonREPLTool()
tools = [python_repl_tool, sql_query_tool]

# tools = load_tools(['Python_REPL'])
# tools.append(sql_query_tool)


# Specify the tools the agent may use
tool_names = [tool.name for tool in tools]



# Initialize st.session_state if it's not already initialized
if not hasattr(st, "session_state"):
    st.session_state = {}


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
#     # Your code to append messages
#     st.session_state["messages"].append({"role": "system", "content": get_system_prompt()})

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": get_system_prompt()}]



SUFFIX_SYSTEM_PROMPT = get_system_prompt()


FINAL_INPUT = """
"Request: {input}"
"{agent_scratchpad}"
"""

SUFFIX_SYSTEM_PROMPT += FINAL_INPUT

prompt = ZeroShotAgent.create_prompt(
    tools, prefix=PREFIX, suffix=SUFFIX_SYSTEM_PROMPT, input_variables=["input", "agent_scratchpad"]
)
# llm_with_stop = llm.bind(stop=["\nObservation"])

llm_chain = LLMChain(llm=llm, prompt=prompt)


# Specify the tools the agent may use
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(memory_key="chat_history")

# agent_executor = initialize_agent(
#     tools,
#     llm_chain,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
#     memory=memory
# )


# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


# # Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create a dictionary with 'input' and 'context'
    input_dict = {
        'input': prompt,
        'context': SUFFIX_SYSTEM_PROMPT,
        'handle_parsing_errors': True}

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(**input_dict)
        st.write(response)
