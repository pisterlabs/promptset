from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url , choose_first_url


def lookup(searched_name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """
        Given the full name {name_of_person} , I want you to get it me a link to their Linkedin profile.
        Your answer should contain only URL. Do not add here sure etc. Just give the URL.
    """

    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url,
            description="useful when you need get the Linkedin Page URL",
        ),
        Tool(
            name="Choose one of the LinkedIn'urls.",
            func=choose_first_url,
            description="useful when you have multiple LinkedIn url's",
        ),
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt = PromptTemplate(template=template, input_variables=["name_of_person"])

    linkedin_profile_url = agent.run(prompt.format_prompt(name_of_person=searched_name))
    return linkedin_profile_url
