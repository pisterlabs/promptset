from langchain.agents import initialize_agent, AgentType

from langchain.chat_models import ChatOpenAI

from coder.vectorstore import VectorStore


# TODO reimplement react agent - the current one fails to properly handle incomplete scratchpad results
def agent(question: str, vectorstore_collections: list[str]):
    v = VectorStore()
    tools = []
    for collection in vectorstore_collections:
        tools.append(v.get_tool(collection))

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    # llm = ChatOpenAI(temperature=0)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent.run(question)
