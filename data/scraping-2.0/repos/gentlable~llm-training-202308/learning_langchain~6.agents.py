from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
import langchain


langchain.verbose = True

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["Search"], llm=chat)
agent_executor = initialize_agent(
    tools, chat, agent="zero-shot-react-description")

result = agent_executor.run("please search for a cat")
print(result)
