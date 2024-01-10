from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema.language_model import BaseLanguageModel

from tools.serpapi import search


def get_linkedin_profile_url(llm: BaseLanguageModel, name: str) -> str:
    """
    Given a name of a person, returns their LinkedIn profile url
    :param llm: Base language model to use for COT prompting
    :param name: Name of a person to search
    :return: url
    """
    template = """
    Given a person name {name}, I want you to get me the link of their LinkedIn profile page
    Your answer should only contain a url.
    """

    tools_for_agent = [
        Tool(
            name="Search Google for LinkedIn profile page",
            func=search,
            description="Use this tool, when you need to find the LinkedIn profile page url",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(input_variables=["name"], template=template)
    return agent.run(prompt_template.format_prompt(name=name))
