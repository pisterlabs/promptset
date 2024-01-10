
#!/usr/bin/python3
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
import os
import cgi
my_api_key = "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"
os.environ['OPENAI_API_KEY'] = my_api_key
agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    "ravi.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
form = cgi.FieldStorage()
user_input = form.getvalue('userInput')
print("content-type: text/html")
print()
output = agent.run(user_input)
print(output)