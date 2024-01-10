from langchain import LLMMathChain, SQLDatabase
from langchain.agents import create_csv_agent, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory

from .config import AppConfig, setup_logging

logger = setup_logging()


def build_conversational_retrieval_chain(
    config: AppConfig, top_k: int = 3, combine_chain_type: str = "stuff"
):
    # Vector store retriever
    store = config.get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": top_k})

    # LLM
    llm = config.get_llm()

    # Memory - placeholder to be passed in during inferencing
    memory = ReadOnlySharedMemory(
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    return ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT),
        combine_docs_chain=load_qa_with_sources_chain(
            llm=llm, chain_type=combine_chain_type
        ),
        memory=memory,
        verbose=True,
    )


def build_math_chain(config: AppConfig):
    llm = config.get_llm()
    return LLMMathChain.from_llm(llm=llm, verbose=True)


def build_sql_database_chain(config: AppConfig, db_uri: str):
    llm = config.get_llm()
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True
    )


def build_csv_chain(config: AppConfig, csv_path: str):
    llm = config.get_llm()
    return create_csv_agent(
        llm=llm,
        path=csv_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix="Remove any ` from the Action Input",
    )
