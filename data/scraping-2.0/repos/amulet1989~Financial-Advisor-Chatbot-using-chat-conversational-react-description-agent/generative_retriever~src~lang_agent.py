from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentExecutor
from src import settings
from typing import List
from src.utils import make_retriever


def setup_llm() -> ChatOpenAI:
    """Creates a ChatOpenAI instance

    Returns:
        ChatOpenAI
    """
    llm = ChatOpenAI(
        openai_api_key=settings.API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,
    )
    return llm


def setup_tools(llm: ChatOpenAI) -> List[Tool]:
    """Creates a list of tools to be used in an agent

    Args:
        llm (ChatOpenAI): chat model to be used in tools and agent

    Returns:
        List[Tool]
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=make_retriever()
    )
    tools = [
        Tool(
            name="BM25 Retrieval",
            func=qa.run,
            description=(
                "useful to ask for information. Try using relevant keywords for a BM25 algorithm. "
                "Do not omit important input details such as dates, numbers, company names or acronyms; "
                "especially if the acronyms are written between parenthesis, for example: (DOO)."
            ),
        ),
    ]
    return tools


def make_agent() -> AgentExecutor:
    """Initializes a chat conversational agent with a chat model and a RetrievalQA tool

    Returns:
        AgentExecutor
    """
    llm = setup_llm()
    tools = setup_tools(llm)
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=2,
        early_stopping_method="generate",
        agent_kwargs={
            "stop": ["\nObservation:"],
        },
        handle_parsing_errors=True,
        verbose=True
    )
    return agent
