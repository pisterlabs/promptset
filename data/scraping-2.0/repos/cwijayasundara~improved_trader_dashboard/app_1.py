from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo",
                 temperature=0,
                 callbacks=[StreamingStdOutCallbackHandler()],
                 streaming=True)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

agent.run("Analyze apple stock and craft investment recommendations")
