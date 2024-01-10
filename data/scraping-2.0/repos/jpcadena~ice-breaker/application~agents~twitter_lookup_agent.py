"""
A module for twitter lookup agent in the application-agents package.
"""
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from application.utils.custom_serp_api_wrapper import get_profile_url


def lookup(name: str) -> str:
    """
    Lookup a Twitter profile in the application
    :param name: The name of the profile to lookup
    :type name: str
    :return: The profile from Twitter
    :rtype: str
    """
    llm: ChatOpenAI = ChatOpenAI(temperature=0)
    template: str = """
       given the name {name_of_person} I want you to find a link to their
        Twitter profile page, and extract from it their username
       In Your Final answer only the person's username"""
    tools_for_agent_twitter: list[Tool] = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url,
            description="useful for when you need get the Twitter Page URL",
        ),
    ]
    agent: AgentExecutor = initialize_agent(
        tools_for_agent_twitter,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template: PromptTemplate = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    twitter_username: str = agent.run(
        prompt_template.format_prompt(name_of_person=name)
    )
    return twitter_username
