from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.search_tool import get_uid


def lookup_v(flower_type: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name='gpt-4-1106-preview')
    template = """
    Given the {flower} I want you to get a related 微博 UID.
    Your answer should contain only a UID.
    The URL always starts with https://weibo.com/u/ 
    For example, if https://weibo.com/u/1669879400 is her 微博, then 1669879400 is her UID 
    This is only the example don't give me this, but the actual UID"""
    prompt_template = PromptTemplate.from_template(template=template)
    tools = [
        Tool.from_function(
            func=get_uid,
            name='Crawl Google for 微博 page',
            description="useful for when you need get the 微博 UID",
        ),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent.run(prompt_template.format_prompt(flower=flower_type))
