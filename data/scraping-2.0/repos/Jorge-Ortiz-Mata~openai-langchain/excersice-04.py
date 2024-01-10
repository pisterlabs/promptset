from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv

load_dotenv()

# Agents excersice
def langchain_agent():
  llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")

  tools = load_tools(["wikipedia", "llm-math"], llm = llm)

  agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
  )

  response = agent.run(
    "What is the averege age of dogs? Multiply by 4"
  )

  print(response)

langchain_agent()
