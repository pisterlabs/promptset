"""Chatlas Agent for workin with the Pandas DF."""

import pandas as pd
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonAstREPLTool

from chatlas.prompts.prompts_df import PREFIX, SUFFIX


def create_chatlas(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    prefix = PREFIX
    suffix = SUFFIX
    number_of_head_rows = 5
    callback_manager = None

    # Setup input variables for the filling in the prompt
    input_variables = ["input", "agent_scratchpad"]
    input_variables += ["chat_history"]  # for using memory
    input_variables += ["df_head"]  # for adding dataframe sample to the prompt

    # Create tools
    tools = [PythonAstREPLTool(locals={"df": df})]
    tool_names = [tool.name for tool in tools]

    # Create prompts
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=input_variables)
    prompt = prompt.partial()
    prompt = prompt.partial(df_head=str(df.head(number_of_head_rows).to_markdown()))  # add df sample to the prompt

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Boot up zero-shot agent with LLMChain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
    )

    # Setup agent executor (router for agent)
    agent_exec = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        callback_manager=callback_manager,
        verbose=True,
        return_intermediate_steps=False,
        max_iterations=15,
        max_execution_time=None,
        early_stopping_method="force",
        handle_parsing_errors=True,
    )

    return agent_exec
