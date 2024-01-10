from pathlib import Path
import os

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
import dotenv

from lang_agency import tools, memory, chains


# llm
dotenv_file = dotenv.find_dotenv(str(Path("./").absolute().joinpath(".env")))
dotenv.load_dotenv(dotenv_file)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# agent
agent_chain = initialize_agent(
    llm=llm,
    tools=tools.tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs=memory.agent_kwargs,
    memory=memory.memory,
    # prompt="",
    max_iterations=10,
    # max_execution_time=5,
    verbose=True,
    handle_parsing_errors=True,
    # return_intermediate_steps=True,
    early_stopping_method="generate",
)

def chatbot(inputs: str) -> str:
    answer = agent_chain.run(input=inputs)
    return chains.conversation_chain.predict(input=answer)