from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=None)
    # 군더더기 없는 응답을 받기위해 answer가 무엇인지를 명확히한다.
    template = """given the full name {name_of_person}, I want you to get it me a link to their Linkedin profile page.
                Your answer should contain only a URL."""

    # Tool 마다 name은 required이며 고유해야한다.
    # function도 required 이며 LLM이 사용하기로 결정했을 경우 호출되는 함수다.
    # description은 툴에대한 설명이 명확하고 구체적이어야 예상치 못한 행동을하지 않는다.
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    # agent type은 작업 수행 시 어떤 추론과정을 따를지 결정하므로 매우 중요한 매개변수다.
    # verbose parameter는 agent의 추론과정을 볼 수 있고, 해당 task에 대해 자세한 이해를 할 수 있게해준다.
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))

    protocol = "https"
    prev, url = linkedin_profile_url.split(protocol)
    if prev:
        if url.endswith("."):
            url = url[:-1]
        linkedin_profile_url = protocol + url

    return linkedin_profile_url
