# Agents
# Agents are the decision makers that can look at data, reason about what the next action should be, and execute that action via tools.
# Use cases: Run programs autonomously without the need for human input.
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Helpers
import json

from langchain.llms import OpenAI 

# Agent imports
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# Tool imports
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import TextRequestsWrapper

# For this example, we will pull Google search results. Useful if one needs a list of websites for a research project.

# https://programmablesearchengine.google.com/controlpanel/create
# https://console.cloud.google.com/apis/credentials?pli=1&project=terraform-gcp-example-377014


# os.environ["GOOGLE_CSE_ID"] 
google_cse_id = os.getenv("GOOGLE_CSE_ID")
# os.environ["GOOGLE_API_KEY"] 
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Initialise both the tools you'll be using. For this example, we'll search Google and also give the LLM the ability to execute Python code.
search = GoogleSearchAPIWrapper(google_cse_id=google_cse_id, google_api_key=google_api_key)
requests = TextRequestsWrapper()

# Put both tools in a toolkit
toolkit = [
    Tool.from_function(
        name = "Search",
        func = search.run,
        description = "useful for when you need to search Google to answer questions about current events"
    ),
    Tool.from_function(
        name = "Requests",
        func = requests.get,
        description="Useful for when you want to make a request to a URL"
    )
]

# Create your agent by giving it the tools, LLM and the type of agent that it should be
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

# Time for a question: give it one that it should go to Google for
response = agent({"input":"What is the capital of canada?"})
print (response['output'])

# Time for a question that requires listing the current directory
# response = agent({"input":"Tell me what the comments are about on this webpage https://news.ycombinator.com/item?id=34425779"})
# print (response['output'])