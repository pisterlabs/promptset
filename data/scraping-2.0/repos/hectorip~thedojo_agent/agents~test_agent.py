# Creando un agente con LangChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

print("Este agente puede buscar en Google y hacer aritmética básica.")

while True:
    query = input("Consulta: ")
    if not query:
        break
    print(agent.run(query))
