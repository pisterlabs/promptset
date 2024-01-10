from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url

load_dotenv()


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore
    template = """
        Given the full name {name_of_person} I want you to find a link to their Twitter profile page,
        and extract from it their Twitter username.
        In your final answer only include the Twitter username.
    """
    tool_for_agent = [
        Tool(
            name="Crawl Google for Twitter profile page",
            func=get_profile_url,
            description="useful when you need to know the Twitter username of a person",
        )
    ]
    agent = initialize_agent(
        tools=tool_for_agent,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))

    return twitter_username
