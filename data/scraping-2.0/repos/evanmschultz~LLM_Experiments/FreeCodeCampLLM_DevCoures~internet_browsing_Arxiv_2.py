from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import os
import chainlit
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os


@chainlit.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools: list = load_tools(["arxiv"])

    agent_chain: AgentExecutor = initialize_agent(
        tools=tools,
        llm=llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    chainlit.user_session.set("agent", agent_chain)


@chainlit.on_message
async def main(message):
    agent: AgentExecutor = chainlit.user_session.get("agent")  # type: ignore
    callbacks = chainlit.LangchainCallbackHandler(stream_final_answer=True)

    await chainlit.make_async(agent.run)(message, callbacks=[callbacks])
