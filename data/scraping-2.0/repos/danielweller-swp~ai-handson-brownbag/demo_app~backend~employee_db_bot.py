from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import BaseTool
import json
from uuid import UUID
from azure_openai import setupAzureOpenAI

gpt35 = setupAzureOpenAI()


employeeDb = [ 
  {
    "employeeId": "c75f0c1b-eba1-41c3-9438-5466a2e7d164",
    "name": "Daniel Weller",
    "supervisor": "fd81ff0f-04a4-4c8e-9c79-bfe0e6aae30f",
    "role": "Software Developer",
    "email": "daniel.weller@trustbit.tech"
  },
  {
    "employeeId": "fd81ff0f-04a4-4c8e-9c79-bfe0e6aae30f",
    "name": "Seb Burgstaller",
    "supervisor": "006cd43a-8852-42e4-95ff-c09a929792de",
    "role": "CTO",
    "email": "seb.burgstaller@trustbit.tech"    
  },
  {
    "employeeId": "006cd43a-8852-42e4-95ff-c09a929792de",
    "name": "Jörg Egretzberger",
    "supervisor": None,
    "role": "CEO",
    "email": "office@trustbit.tech"    
  }
]

def is_uuid(str):
   try:
      UUID(str)
      return True
   except ValueError:
      return False


def search_employee_db(query):
  if is_uuid(query):
    result = [e for e in employeeDb if e["employeeId"] == query]
  else:
    result = [e for e in employeeDb if e["name"] == query]

  if (len(result) > 0):
      return result[0]
  else:
      return None


class EmployeeDatabaseTool(BaseTool):
    name = "Trustbit Employee Database"
    description = """Useful for when you need to answer questions about Trustbit employees. 
                    Provide either the employee name or the employee id (without quotes) 
                    as action input."""

    def _run(self, query: str) -> str:
        employee = search_employee_db(query)

        return json.dumps(employee)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


def get_employee_db_completion(msg, retries=0):
  query = f"My employee id is c75f0c1b-eba1-41c3-9438-5466a2e7d164. Query: {msg}"
  tools = [EmployeeDatabaseTool()]

  # https://python.langchain.com/en/latest/index.html
  agent = initialize_agent(tools, 
                           llm=gpt35, 
                           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                           verbose=True)
  try:
    result = agent.run(query)
    return result
  except Exception as e:
     if retries < 5:
        return get_employee_db_completion(msg, retries+1)
     else:
        raise e