from elevenlabs import clone, generate, play, set_api_key
from elevenlabs.api import History
import ffmpeg
import subprocess
import sys
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType, load_tools 
from langchain.utilities import WikipediaAPIWrapper, PythonREPL, TextRequestsWrapper, GoogleSearchAPIWrapper
from langchain.tools import ShellTool
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit, GmailToolkit
from tempfile import TemporaryDirectory
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
set_api_key('2c020018e5faf6849f2dbae409d5f310')
os.environ["GOOGLE_CSE_ID"] = "9257a94d07b114416"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBfMrZ2bkFbh-CGIMM2zHzROJQevxNSaVs"
OPENAI_API_KEY = "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"
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)
def test(prompt):
    python_repl = PythonREPL()
    requests_tools = load_tools(["requests_all"])
    # Each tool wrapps a requests wrapper
    requests_tools[0].requests_wrapper
    TextRequestsWrapper(headers=None, aiosession=None)
    requests = TextRequestsWrapper()

    # We'll make a temporary directory to avoid clutter
    working_directory = TemporaryDirectory()
    filetoolkit = FileManagementToolkit(root_dir=str(working_directory.name), selected_tools=["read_file", "write_file", "list_directory"]).get_tools()
    read_tool, write_tool, list_tool = filetoolkit
    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file='token.json',
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
        )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit() 
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)
 
    wikipedia = WikipediaAPIWrapper()
    shell_tool = ShellTool()
    search= GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to get information from wikipedia about a single topic"
        ),
        Tool(
            name="ShellTool",
            func=shell_tool.run,
            description="Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command."
        ),
         Tool(
            name="list_directory",
            func=list_tool.run,
            description="Interact with the local file system. List the files in a directory."
        ),
        Tool(
            name="python_rep1",
            func=python_repl.run,
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
        ),
#         Tool(
#            name="requests",
#            func=requests.get,
#            description="A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page."
#        ),
        Tool(
            name="Gmail Toolkit",
            func=agent.run,
            description="Useful for sending, reading, writing, viewing emails from the Gmail API."
        ),
    ]

    agent_executor = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True, handle_parsing_errors=True)
    output = agent_executor.run(prompt)
    return output
def main():

    # Read user input from standard input
    user_input = sys.stdin.readline().rstrip()
    prompt = user_input
    bot_response = str(test(prompt))
    # Write the bot's response to standard output
    sys.stdout.write(bot_response)
    sys.stdout.flush()
if __name__ == '__main__':
    main()



