"""
A module for LinkedIn lookup agent in the application-agents package.
"""
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from application.utils.custom_serp_api_wrapper import get_profile_url


def lookup(name: str) -> str:
    """
    Lookup a LinkedIn profile in the application
    :param name: The name of the profile to lookup
    :type name: str
    :return: The profile from LinkedIn
    :rtype: str
    """
    llm: ChatOpenAI = ChatOpenAI(temperature=0)
    template: str = """given the full name {name_of_person} I want you to get 
    it me a link to their Linkedin profile page.
      Your answer should contain only a URL"""
    tools_for_agent1: list[Tool] = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        ),
    ]
    agent: AgentExecutor = initialize_agent(
        tools_for_agent1, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    prompt_template: PromptTemplate = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    linkedin_username: str = agent.run(
        prompt_template.format_prompt(name_of_person=name)
    )
    return linkedin_username
