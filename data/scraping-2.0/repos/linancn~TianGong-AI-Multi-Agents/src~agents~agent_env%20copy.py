import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.render import format_tool_to_openai_function
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent


load_dotenv()


def env_agent():
    tools = [DuckDuckGoSearchRun()]

    chat_model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        verbose=True,
        streaming=True,
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent = initialize_agent(
        tools,
        llm=chat_model,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

    return agent
