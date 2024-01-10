"""
This script integrates various components from the langchain library to create an 
AI agent capable of interacting with an SQLite database. It demonstrates how to set 
up a chat-based agent that can understand and respond to queries about the database, 
utilizing custom tools for running SQL queries and describing table schemas, and a 
custom handler for message formatting.

The script utilizes the OpenAIFunctionsAgent from langchain, along with custom tools 
for database interaction. The agent is configured with a specific prompt template and 
is enhanced with a conversation memory buffer, allowing it to maintain context over 
multiple interactions. It is capable of executing SQL queries and describing database tables.

A custom message handler, ChatModelStartHandler, is used to format and display messages 
during the conversation, enhancing the readability and interaction experience.

Functions:
- list_tables: Retrieves the list of table names from the SQLite database.
- run_sqlite_query: Executes an SQL query against the SQLite database and returns the results.
- describe_tables: Provides the schema of specified tables in the SQLite database.

Classes:
- RunQueryArgsSchema: Pydantic schema for the 'run_sqlite_query' function arguments.
- DescribeTablesArgsSchema: Pydantic schema for the 'describe_tables' function arguments.
- ConversationBufferMemory: A memory buffer that retains the conversation history for the agent.
- ChatModelStartHandler: Custom handler for formatting and 
displaying chat messages during the conversation.

Usage:
The script is designed to be executed as a standalone module. Upon running, it initializes 
an agent capable of processing natural language queries related to the database and executing 
corresponding SQL commands. The conversation memory buffer aids in maintaining context 
across multiple queries, and the ChatModelStartHandler enhances the presentation of 
the chat interactions.
"""

from dotenv import load_dotenv
from handlers.chat_model_start_handler import ChatModelStartHandler
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from tools.report import write_report_tool
from tools.sql import describe_tables_tool, list_tables, run_query_tool

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])

# Retrieve the list of tables in the SQLite database
TABLES = list_tables()

# Define the chat prompt template with system message and placeholders
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to SQLITE database.\n"
                f"The database has tables of: {TABLES}"
                "Do not make any assumptions about what tables exist "
                "or what columns exists. Instead, use the 'describe_tables' function"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Initialize a conversation memory buffer to retain context over multiple interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define tools for running SQL queries, describing table schemas, and writing reports
tools = [run_query_tool, describe_tables_tool, write_report_tool]

# Create an OpenAIFunctionsAgent with the defined prompt, tools, and conversation memory
agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

# Create an AgentExecutor to execute the agent without verbose logging and conversation memory
agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory,
)

# Execute the agent to answer queries and potentially write reports
agent_executor("How many orders are there? Write the result to an html report.")
agent_executor("Repeat the exact same process for users.")
