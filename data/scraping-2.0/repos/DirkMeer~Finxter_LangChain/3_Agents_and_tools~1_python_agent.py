from decouple import config
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool


chat_gpt_api = ChatOpenAI(
    openai_api_key=config("OPENAI_API_KEY"),
    temperature=0,
    model="gpt-3.5-turbo-0613",
)

agent = create_python_agent(
    llm=chat_gpt_api,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)


agent.run(
    "Please write a function to calculate which weekday (monday, tuesday, etc.) a given date is. The date should be in format 'YYYY-MM-DD' and the function should return a string with the weekday name. Return the weekday for the date '2025-06-13'."
)
