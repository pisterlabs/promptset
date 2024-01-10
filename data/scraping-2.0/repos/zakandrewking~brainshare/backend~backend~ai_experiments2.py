from typing import Final

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain.tools.render import format_tool_to_openai_function

from backend import models

# # Do this so we can see exactly what's going on under the hood
# from langchain.globals import set_debug
# set_debug(True)


async def chat_with_tools(query: str, session: AsyncSession, user_id: str) -> tuple[str, int]:
    input_formatter = {
        "query": lambda x: x["query"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant designed to output JSON."),
            ("user", "{query}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    @tool
    async def find_user_files() -> str:
        """Finds files owned by the user."""
        res = list(
            (
                await session.execute(
                    select(models.FileData).where(models.FileData.user_id == user_id)
                )
            ).scalars()
        )
        return "\n".join(f"file with ID {f.id}" for f in res)

    tools = [find_user_files]

    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
    ).bind(
        functions=[format_tool_to_openai_function(t) for t in tools],
    )

    agent = input_formatter | prompt | llm | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # good example for openai multi tool, for when you want to run multiple
    # tools at once
    # https://github.com/langchain-ai/langchain/issues/8325
    # let's stay serial and use the OPENAI_FUNCTIONS (which i think is the same
    # as the above?)

    res = await agent_executor.ainvoke({"query": query, "agent_scratchpad": []})

    return str(res["output"]), 0


# # NOTE: Using JSON mode with LCEL:
# # you need to instruct the llm to output JSON
# sysmsg = SystemMessage(content="You are a helpful assistant designed to output JSON.")
# # shortcut for HumanMessagePromptTemplate.from_template("{query}")
# prompt = sysmsg + {query}"
# # use a newish version of gpt4
# model = ChatOpenAI(
#     model_name="gpt-4-1106-preview",
# # add the options
# ).bind(response_format={ "type": "json_object" })
# # run it
# chain = prompt | model
# res = chain.invoke({"query": query})
# # res.content is a JSON string
# return str(res.content)
