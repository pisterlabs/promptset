from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def linkedin_lookup_agent(name: str, reference: dict) -> str:
    """Lookup a LinkedIn URL based on a name"""
    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-4",
    )

    template = """
      I want you to get the LinkedIn URL of {name_of_person}. This person have worked at {companies} and studied at {education}. Your answer should contain only a URL to the LinkedIn profile page.
    """

    tools_for_agent = [
        Tool(
            name="Crawl Google for Linkedin profile page",
            func=get_profile_url,
            description="Useful for when you need to get a LinkedIn profile page",
        )
    ]

    agent = initialize_agent(
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools_for_agent,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        input_variables=["name_of_person", "companies", "education"],
        template=template,
    )

    linkedin_profile_url = agent.run(
        prompt_template.format_prompt(
            name_of_person=name,
            companies=reference["companies"],
            education=reference["education"],
        )
    )

    return linkedin_profile_url
