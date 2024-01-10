# requerimento especifico daqui: google-search-results

from maritalkllm import MariTalkLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import SerpAPIWrapper, WikipediaAPIWrapper

if __name__ == "__main__":
    g_key = "757f89415e73a61fcf733e8cbcd997c91bd472a5fb72bf7de19afe5d739bdee4"

    prompt = """Quais os assuntos abordados sobre evolução humana na internet e na wikipedia? 
    Depois disso, liste só alguns dos principais tópicos encontrados."""

    model = MariTalkLLM()
    g_param = {
        "engine": "google",
        "gl": "br",
        "google_domain": "google.com",
        "hl": "pt",
    }
    search = SerpAPIWrapper(params=g_param, serpapi_api_key=g_key)  # type: ignore
    wikipedia = WikipediaAPIWrapper(lang="pt", top_k_results=2)  # type: ignore
    tools = [
        Tool.from_function(
            func=wikipedia.run,
            name="Wikipedia",
            description="Use para pesquisar informações só na wikipedia",
        ),
        Tool.from_function(
            func=search.run,
            name="Search",
            description="Use para pesquisar informações só da internet",
        ),
    ]
    agent = initialize_agent(
        tools,
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    print(f"Test Agent: {prompt}")
    agent.run(prompt)
