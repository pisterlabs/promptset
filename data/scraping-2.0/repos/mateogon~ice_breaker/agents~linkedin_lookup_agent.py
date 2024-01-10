from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import Together
import os
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = Together(
        model="DiscoResearch/DiscoLM-mixtral-8x7b-v2",
        temperature=0,
        max_tokens=512,
        top_k=50,
        top_p=1,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
    )
    template = """
        given the full name {name_of_person} I want you to give me a link to their Linkedin profile page. Your answer should contain only a URL.
        """
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the linkedin url of a person",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
