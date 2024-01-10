from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given a full name {name_of_person} I want you to get me a link to their LinkedIn profile page. 
                Your answer should contain only a URL"""

    tool_for_agent = [
        Tool(
            name="Crawl Google for Linkedin Profile Page",
            func=get_profile_url,
            description="Useful when you need to get the linkedin page URL",
        )
    ]

    agent = initialize_agent(
        tools=tool_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        stop=["\nObservation"],
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))

    return linkedin_profile_url
