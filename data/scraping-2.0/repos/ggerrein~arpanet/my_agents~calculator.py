from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, LLMMathChain
from langchain.chat_models import ChatOpenAI

def get_tool():
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
    chain = LLMMathChain.from_llm(llm, verbose=True)
    tool = Tool(
        name="Search",
        func=chain.run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    )
    return tool