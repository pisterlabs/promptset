from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.agents import create_csv_agent





OPENAI_API_KEY = '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'


agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    "2023-9-20-eltiempo.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("how many rows are there?")