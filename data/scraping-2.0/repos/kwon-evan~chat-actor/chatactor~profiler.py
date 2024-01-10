import os
import json

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.tools import WikipediaQueryRun


def _build_tools() -> list[Tool]:
    search = DuckDuckGoSearchAPIWrapper(region="kr-kr")
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ko"))
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you want to search for something on the internet.",
        ),
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Useful for when you want to search for something on Wikipedia.",
        ),
    ]
    return tools


def _build_prompt(tools: list[Tool] | None = None) -> PromptTemplate:
    prefix = """
    You are a Entertainment Reporter. 
    You have been tasked to profile a given person.
    Do not use your own knowledge, use the internet to find the information.

    You have access to the following tools:"""

    suffix = """Begin! Remember to use the tools to find the information.
    The Final Answer is in json format, and not key, values must be written in Korean. as follows:
    You have to find the following information about the person:
    name: the person's name
    occupation: the person's occupation
    birth: the birth of the person formatted in YYYY-MM-DD
    death: the death of the person formatted in YYYY-MM-DD
    summary: the summary of the person in 1-2 sentences

    Person: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools if tools is not None else _build_tools(),
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )

    return prompt


class Profiler(ZeroShotAgent):
    # TODO: Implement this
    # Profiler.run() should save the output to a json file
    pass


def get_profiler(
    tools: list[Tool] | None = None,
    prompt: PromptTemplate | None = None,
    openai_api_key: str | None = None,
) -> AgentExecutor:
    if tools is None:
        tools = _build_tools()

    if prompt is None:
        prompt = _build_prompt()

    if openai_api_key is None:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set.")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=openai_api_key
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    return agent_executor


if __name__ == "__main__":
    tools = _build_tools()
    prompt = _build_prompt()
    profiler = get_profiler(tools=tools, prompt=prompt)

    output = profiler.run("일론 머스크")
    person = json.loads(output)  # json str -> dict

    with open(f"profiles/{person['name']}.json", "w") as f:
        json.dump(person, f, indent=2, ensure_ascii=False)  # dict -> save as json file
