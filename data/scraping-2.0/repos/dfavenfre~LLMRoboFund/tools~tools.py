from Helpers.helper_functions import filter_embeddings, get_schema, run_query, save
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_sql_agent
from langchain.chat_models import ChatOpenAI, ChatCohere
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.sql_database import SQLDatabase
from langchain.chains import RetrievalQA
from langchain.agents import AgentType
import pinecone
import json
import os


def query_from_sql():
    """
    Use this tool when you need to fetch some data from SQL database to answer\
    questions related to monthly and annual returns of funds,\
    management fees, number of initial outstanding shares, and etc.
    """
    sql_prompt = ""
    with open("Prompts/sql_agent_prompts.json", "r") as file:
        sql_prompt = json.load(file)

    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=1e-10,
        model="gpt-3.5-turbo",
    )

    database = SQLDatabase.from_uri(database_uri=os.environ.get("uri_path"))
    toolkit = SQLDatabaseToolkit(
        db=database,
        llm=llm,
    )
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        early_stopping_method="force",
        suffix=str(sql_prompt),
        max_iterations=10,
        verbose=True,
    )

    return sql_agent


def query_from_vdb(text: str) -> str:
    """
    Use this tool when you need to answer questions related to a funds' financial risks,\
    investment strategy and the invested financial instruments.\
    Also you can answer general information about funds, such as the managing company.
    """

    chat_model = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo",
    )
    embeddings_cohere = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-ada-002"
    )
    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),
        environment=os.environ.get("pinecone_environment_value"),
    )

    searcher = Pinecone.from_existing_index(
        index_name=os.environ.get("pinecone_index"), embedding=embeddings_cohere
    )
    context_compressor_retriever = filter_embeddings(
        search_object=searcher,
        embedding_model=embeddings_cohere,
        s_threshold=0.6,
        r_threshold=0.8,
    )

    chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="refine",
        retriever=context_compressor_retriever,
    )
    return chain.run(text)


def query_sql(question: str):
    db = SQLDatabase.from_uri(os.environ.get("uri_path"))

    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo",
    )

    template = """Based on the table schema below, write a SQL query that would answer the user's question. The output you will generate must contain only the SQL query statement, do not write anything else as the answer, just the sql statement and don't ask if the user wants anything else. Your answer MUST be only the sql statement, and DO NOT EVER write anythin else
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question, convert it to a SQL query. No pre-amble.",
            ),
            ("human", "What is the average monthly return"),
            (
                "ai",
                "'SELECT AVG(monthly_return) AS avg_monthly_return FROM tefastable;'",
            ),
            ("human", "get me the top 5 funds with the highest monthly return"),
            ("ai", "'SELECT monthly_return FROM tefastable DESC LIMIT 5;'"),
            ("human", template),
        ]
    )

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    sql_response.invoke({"question": question})

    # Chain to answer
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question and SQL response, convert it to a natural langugae answer. No pre-amble.",
            ),
            ("human", template),
        ]
    )

    full_chain = (
        RunnablePassthrough.assign(query=sql_response)
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | llm
    )

    full_chain.invoke({"question": question})

    template = """Given an input question, convert it to a SQL query. No pre-amble. Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    memory = ConversationBufferMemory(return_messages=True)

    # Chain to query with memory
    sql_chain = (
        RunnablePassthrough.assign(
            schema=get_schema,
            history=RunnableLambda(
                lambda x: memory.load_memory_variables(x)["history"]
            ),
        )
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save
    sql_response_memory.invoke({"question": "What is the average monthly return"})

    # Chain to answer
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question and SQL response, convert it to a natural langugae answer. No pre-amble.",
            ),
            ("human", template),
        ]
    )

    full_chain = (
        RunnablePassthrough.assign(query=sql_response_memory)
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | llm
    )

    return full_chain.invoke({"question": question})
