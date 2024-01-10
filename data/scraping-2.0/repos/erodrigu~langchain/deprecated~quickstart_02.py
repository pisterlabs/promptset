from langchain.llms import Ollama
from langchain.schema import HumanMessage, AIMessage, SystemMessage


llm = Ollama(model="orca-mini")


text = "What is a 2x4 in the context of lumber?"
messages = [("human",[HumanMessage(content=text)]), ("AI", [AIMessage(content=text)]), ("System", [SystemMessage(content=text)])]


for role, message in messages:
    print(f"input role {role}, {llm.predict_messages(message, temperature=0)}\n")
    



