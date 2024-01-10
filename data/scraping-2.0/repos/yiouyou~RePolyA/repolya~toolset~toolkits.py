from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents.agent_toolkits import O365Toolkit

from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

from langchain.agents.agent_toolkits.github.toolkit import GitHubToolkit
from langchain.utilities.github import GitHubAPIWrapper

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool


gmail_toolkit = GmailToolkit()
gmail_tools = gmail_toolkit.get_tools()

O365_toolkit = O365Toolkit()
O365_tools = O365_toolkit.get_tools()

# sql_db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
# sql_toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

# github_toolkit = GitHubToolkit.from_github_api_wrapper(GitHubAPIWrapper())

# agent_executor = create_python_agent(
#     llm=OpenAI(temperature=0, max_tokens=1000),
#     tool=PythonREPLTool(),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

