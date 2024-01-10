"""Streamlit app generate SQL statements from natural language queries."""
import os
import sys
from typing import Any, Optional, Union
from urllib.parse import quote_plus

import chromadb
import pandas as pd
import sqlalchemy as sa
import streamlit as st
from langchain import OpenAI
from llama_index import (
    Document,
    GPTSQLStructStoreIndex,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SQLDatabase,
)

from llama_index import SimpleDirectoryReader

from llama_index.composability import ComposableGraph
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.common.struct_store.schema import SQLContextContainer
from llama_index.indices.list import GPTListIndex
from llama_index.indices.struct_store import SQLContextContainerBuilder
from llama_index.prompts.prompts import TextToSQLPrompt
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from streamlit.components.v1 import html
from streamlit.logger import get_logger
from streamlit_chat import message

LOGGER = get_logger(__name__)

# from llama_index.readers import Document
RS_TEXT_TO_SQL_TMPL = """You are an AWS Redshift expert. Given an input question,
first create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples
to obtain, query for at most 500 results. You can order the results to return the
most informative data in the database.
You must query only the columns that are needed to answer the question.
Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the folowing tables listed below:
{schema}

Question: {query_str}
SQLQuery: """

RS_TEXT_TO_SQL_PROMPT = TextToSQLPrompt(RS_TEXT_TO_SQL_TMPL, stop_token="\nSQLResult:")


# Function to create a db_engine string
def create_connection_string(dialect: str, host: str, port: int, dbname: str, user: str, password: str) -> str:
    """Create a db_engine string for the database."""
    quote_plus_password = quote_plus(password)
    quote_plus_user = quote_plus(user)
    if dialect == "postgresql":
        return f"{dialect}+psycopg2://{quote_plus_user}:{quote_plus_password }@{host}:{port}/{dbname}"

    return f"{dialect}://{quote_plus_user}:{quote_plus_password }@{host}:{port}/{dbname}"


# Function to connect to the database
@st.cache_resource
def create_db_engine(connection_string: str) -> Any:
    """Connect to the Redshift/Postgres database."""
    return sa.create_engine(connection_string)


@st.cache_resource
def create_llama_db_wrapper(
    _connection: Any, schema: Optional[str] = None, **kwargs: Any
) -> SQLDatabase:
    """Create a SQLDatabase wrapper for the database.

    Args:
        _connection (sa.engine.db_engine): db_engine
        schema (Optional[str], optional): Schema name. Defaults to None.

    Returns:
        SQLDatabase: SQLDatabase wrapper for the database
    """
    return SQLDatabase(_connection, schema=schema)


@st.cache_resource
def build_table_schema_index(
    _sql_database: SQLDatabase, model_name: str, **kwargs: Any
) -> tuple[Any, Any]:
    """Building a table schema index from the SQL database.

    Args:
        _sql_database (SQLDatabase): SQL database
        model_name (str): Model name

    Returns:
        tuple[Any, Any]: table_schema_index, context_builder
    """
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    try:
        # Setup remote connection to Chroma DB
        CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST")
        CHROMA_DB_PORT = os.environ.get("CHROMA_DB_PORT")
        chroma_client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
    except:
        # Otherwise, use in-memory Chroma DB
        print(f"Fail to connect to remote Chroma DB at {CHROMA_DB_HOST}:{CHROMA_DB_PORT}. Use in-memory Chroma DB instead.")
        chroma_client = chromadb.Client()

    chroma_collection = chroma_client.get_or_create_collection("table_schema")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build a vector index from the table schema information
    context_builder = SQLContextContainerBuilder(_sql_database)
    table_schema_index = context_builder.derive_index_from_context(
        GPTVectorStoreIndex,
        service_context=service_context,
        storage_context=storage_context,
    )

    return table_schema_index, context_builder


@st.cache_resource
def build_sql_context_container(
    _context_builder: SQLContextContainerBuilder,
    _index_to_query: Union[BaseGPTIndex, ComposableGraph],
    query_str: str,
    **kwargs: Any,
) -> SQLContextContainer:
    """Build a SQL context container from the table schema index."""
    # query_str
    # kwargs
    query_index_args = {
        "index": _index_to_query,
        "query_str": query_str,
        "store_context_str": True,
    }

    if kwargs.get("dbt_sources_yaml_toggle"):
        query_index_args["query_configs"] = [
            {
                "index_struct_type": "simple_dict",
                "query_mode": "default",
                "query_kwargs": {"similarity_top_k": 3, "verbose": True},
            },
            {
                "index_struct_type": "list",
                "query_mode": "default",
                "query_kwargs": {"verbose": True},
            },
        ]
    else:
        query_index_args["verbose"] = "True"
        query_index_args["similarity_top_k"] = 3

    _context_builder.query_index_for_context(**query_index_args)
    return _context_builder.build_context_container()


@st.cache_resource
def create_sql_struct_store_index(
    _sql_database: SQLDatabase,
    _sql_context_container: SQLContextContainer,
    connection_string: str,
    query_str: str,
) -> GPTSQLStructStoreIndex:
    """Create a SQL structure index from the SQL database.

    Args:
        _sql_database (SQLDatabase): SQL database
        connection_string (str): db_engine connection string for cache invalidation

    Returns:
        GPTSQLStructStoreIndex: SQL structure index
    """
    return GPTSQLStructStoreIndex(
        sql_database=_sql_database, sql_context_container=_sql_context_container
    )


@st.cache_data(ttl=60)
def query_sql_structure_store(
    _index: GPTSQLStructStoreIndex,
    query_str: str,
    **kwargs: Any,
) -> RESPONSE_TYPE:
    """Query the SQL structure index.

    Args:
        _index (GPTSQLStructStoreIndex): SQL structure index
        _sql_context_container (SQLContextContainer): SQL context container
        query_str (str): SQL query string
        openai_api_key (str): OpenAI API key placehoder for cache invalidation

    Returns:
        Dict[str, Any]: Query response
    """
    response = _index.as_query_engine(
        query_mode="nl", text_to_sql_prompt=RS_TEXT_TO_SQL_PROMPT
    ).query(query_str)

    return response


def main() -> int:
    """Start the streamlit app."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # This adds the path of the current file
    # to the PYTHONPATH variable
    sys.path.append(dir_path)

    from js_code_plot_generator import JSCodePlotGenerator
    from rsb import RequestsSummaryBuilder

    st.set_page_config(
        layout="wide",
        page_icon="docs/img/analitiq_logo.png",
        page_title="Analitiq Analyst"
    )

    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)

    st.title("Step 2 - Ask Analitiq")

    st.markdown(
        "Transforming Conversations into Insights.\n"
        "Try asking _Show me my top 10 customers_ or _Give me revenues by month for 2018_"
    )

    # Left pane for db_engine input controls
    with st.sidebar:
        st.image("docs/img/analitiq_logo.png", use_column_width=True)
    
        st.header("Step 1 - Connect to Database")

        st.markdown(
            "This app is connected to a sample e-commerce database of a fictional website, where users buy and sell tickets online for sporting events, shows, and concerts. "
            "More details about the data at [Analitiq website](https://www.analitiq.ai/sample-data-structure/)._"
        )

        db_credentials = st.secrets.get("db_credentials", {})

        with st.expander("Database credentials:"):
            dialect = st.selectbox("Database", ("redshift", "postgresql"))
            host = st.text_input("Host", value=db_credentials.get("host", ""))
            port = st.number_input(
                "Port", min_value=1, max_value=65535, value=db_credentials.get("port", 5439)
            )
            dbname = st.text_input("Database name", value=db_credentials.get("database", ""))
            user = st.text_input("Username", value=db_credentials.get("username", ""))
            password = st.text_input(
                "Password", type="password", value=db_credentials.get("password", "")
            )
            schema = st.text_input(
                "Schema", disabled=True, value=db_credentials.get("schema", "public")
            )

        connect_button = st.button("Connect")

    try:
        # Connect to DB when 'Connect' is clicked
        if not connect_button and not st.session_state.get("connect_clicked"):
            return 1

        st.session_state.connect_clicked = True
        # change label of connect button to 'Connected'
        connection_string = create_connection_string(dialect, host, port, dbname, user, password)
        db_engine = create_db_engine(connection_string=connection_string)

        # Right pane for SQL query input and execution
        if not db_engine:
            return 2

        st.session_state.openai_api_key = st.secrets.get("openai_api_key", st.session_state.get("openai_api_key", ""))

        # openai libs access the key via this environment variable
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

        model_name = "gpt-4"

        cache_invalidation_triggers = {
            "connection_string": connection_string,
            "openai_api_key": st.session_state.openai_api_key,
            "model_name": model_name,
            "schema": str(schema),
        }

        sql_database = create_llama_db_wrapper(db_engine, **cache_invalidation_triggers)

        table_schema_index, context_builder = build_table_schema_index(
            sql_database, **cache_invalidation_triggers
        )

        index_to_query = table_schema_index

        if "is_sample_data_available" not in st.session_state:
            sample_data_docs = SimpleDirectoryReader(input_dir="context").load_data()
            print(f"Loaded {len(sample_data_docs)} documents.")
            metadata_index = GPTListIndex.from_documents(sample_data_docs)
            index_to_query = ComposableGraph.from_indices(
                GPTListIndex,
                [table_schema_index, metadata_index],
                index_summaries=[
                    "Tables' schemas generated via database introspection",
                    "DBT sources yaml",
                ],
            )
            st.session_state.is_sample_data_available = True

        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"]["query"])
                    st.dataframe(message["content"]["dataframe"])
                    with st.expander("JS Plot Code"):
                        st.code(message["content"]["plot_code"], language="javascript")
                    html(message["content"]["plot_code"], scrolling=True, height=500)
            
        if prompt := st.chat_input("Ask Analitiq"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.markdown(":blue[Prepare and execute query...]")

                history_user_content = [
                    x["content"] for x in st.session_state.messages if x["role"] == "user"
                ]
                condensed_query_str = RequestsSummaryBuilder.build_summary(
                    history_user_content,
                    model_name=str(model_name),
                )

                sql_context_container = build_sql_context_container(
                    context_builder,
                    index_to_query,
                    condensed_query_str,
                    **cache_invalidation_triggers,
                )

                try:
                    # cached resource
                    index = create_sql_struct_store_index(
                        sql_database,
                        _sql_context_container=sql_context_container,
                        connection_string=connection_string,
                        query_str=condensed_query_str,
                    )
                    # cached resource
                    response = query_sql_structure_store(
                        _index=index,
                        query_str=condensed_query_str,
                        **cache_invalidation_triggers,
                    )
                except Exception as ex:
                    st.toast(
                        ":red[We couldn't generate a valid SQL query. "
                        "Please try to refine your question with schema, "
                        f"table or column names. Exception info:\n{ex}]"
                    )
                    return 6

                if not response:
                    return 7
                
                sql_query = response.extra_info["sql_query"]
                with st.expander("SQL Code"):
                    st.code(sql_query)

                results = response.extra_info["result"]
                if len(results) == 1 and results[0][0] is None:
                    st.markdown("No data available.")
                else:
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    st.markdown(":blue[Plotting the data. Please wait...]")
                    html_plot_js = JSCodePlotGenerator(sql_query=sql_query, data=df).generate_plot(
                        model_name=str(model_name)
                    )

                    with st.expander("JS Plot Code"):
                        st.code(html_plot_js, language="javascript")

                    html(html_plot_js, scrolling=True, height=500)

                    st.session_state.messages.append({"role": "assistant", "content": {
                            "query": sql_query,
                            "dataframe": df,
                            "plot_code": html_plot_js
                        }
                    })

    except Exception as ex:
        st.markdown(f":red[{ex}]")
        raise ex

    return 0


# Run the Streamlit app
if __name__ == "__main__":
    main()
