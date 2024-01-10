from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate


def lookup(company: str, company_address: str) -> str:
    search = DuckDuckGoSearchRun()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=1000)
    template = """
        Search for the website of {company} at {company_address}
    """

    search_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="A useful tool for searching the internet to find information about companies"
    )

    # Agent
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Prompt template
    prompt_template = PromptTemplate(
        input_variables=["company", "company_address"],
        template=template,
    )

    search_result = agent.run(prompt_template.format_prompt(company=company, company_address=company_address))
    return search_result
