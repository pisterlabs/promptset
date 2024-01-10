import os
import sys
import streamlit as st
import pandas as pd
import json
import base64
from dotenv import load_dotenv
import openai
from llama_index.indices.query.schema import QueryBundle
from brics_tools.apps import logger, log, copy_log


def main():
    # logger.info(f"sys.platform: {sys.platform}")
    if sys.platform != "win32":
        # # these three lines swap the stdlib sqlite3 lib with the pysqlite3 package for chromadb compatibility with streamlit
        # logger.info(
        #     "Swapping stdlib sqlite3 with pysqlite3 for chromadb-linux compatibility"
        # )
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    from brics_tools.utils import helper
    from brics_tools.index_tools.query_engines.studyinfo_query_engine import (
        StudyInfoQueryEngine,
    )
    from brics_tools.index_tools.prompts.studyinfo_prompts import STUDYINFO_QA_PROMPT

    # Page Config
    st.set_page_config(layout="wide")

    # Log: App initialization
    logger.info("Initializing Streamlit app")

    # App title
    st.header("FITBIR Data Repository Study Search w/ RAG", divider="grey")

    # Initialize session state for engine_status if it's not already initialized
    if "engine_status" not in st.session_state:
        st.session_state.engine_status = {"initialized": False}

    # Initialize the engine
    @st.cache_resource(
        show_spinner="Initializing Retriever Engine..."
    )  # cache the engine so it doesn't have to be re-initialized every time
    def initialize_engine():
        logger.info("Initializing query engine")
        cfg = helper.compose_config(
            config_path="../configs/",
            config_name="config_studyinfo",
            overrides=[],
        )
        engine = StudyInfoQueryEngine(cfg)
        engine.init_vector_index()
        engine.create_retriever_only_engine()
        return engine

    # Initialize the engine for the first time, if not already done
    if not st.session_state.engine_status["initialized"]:
        engine = initialize_engine()
        st.session_state.engine_status["initialized"] = True
        logger.info("Query engine initialized")

    # Initialize session state
    if "query_mode" not in st.session_state:
        st.session_state.query_mode = "Retrieval"
    if "query_counter" not in st.session_state:
        st.session_state.query_counter = 0
    if "top_k" not in st.session_state:
        st.session_state.top_k = 10
    if "last_top_k_retriever" not in st.session_state:
        st.session_state.last_top_k_retriever = 10
    if "last_top_k_rag" not in st.session_state:
        st.session_state.last_top_k_rag = 10
    if "top_n_for_llm" not in st.session_state:
        st.session_state.top_n_for_llm = (
            10  # This can be any default value below new_top_k's default
        )
    if "last_top_n_for_llm" not in st.session_state:
        st.session_state.last_top_n_for_llm = st.session_state.top_n_for_llm
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "llm_prompt_text" not in st.session_state:
        st.session_state.llm_prompt_text = STUDYINFO_QA_PROMPT.get_template()
    if "last_llm_prompt_text" not in st.session_state:
        st.session_state.last_llm_prompt_text = STUDYINFO_QA_PROMPT.get_template()
    if "llm_prompt_text_area_key" not in st.session_state:
        st.session_state.llm_prompt_text_area_key = 1
    if "last_llm_model_name" not in st.session_state:
        st.session_state.last_llm_model_name = "gpt-3.5-turbo-16k"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display history
    if st.session_state.history:
        st.header("Query History")
        for item in st.session_state.history:
            query_counter = item["query_counter"]
            st.subheader(f":green[Query: {query_counter}]")
            st.text(item["query"])
            st.subheader(":red[LLM Response:]")
            st.write(item["response"])
            st.subheader("Retrieved Studies:")
            st.dataframe(pd.DataFrame(item["retrieved_studies"]))

    # Sidebar for settings
    st.sidebar.header("Settings", divider="green")

    def set_query_mode():
        if st.session_state.query_mode_radio:
            st.session_state.query_mode = st.session_state.query_mode_radio
            logger.info(f"Setting Query Mode: { st.session_state.query_mode}")

    query_mode = st.sidebar.radio(
        "Query Mode",
        ("Retrieval", "Retrieval-Augmented Generation (RAG)"),
        index=0 if st.session_state.query_mode == "Retrieval" else 1,
        key="query_mode_radio",
        on_change=set_query_mode,
    )

    new_top_k = st.sidebar.slider(
        "Select top_k", 1, 216, st.session_state.top_k, key="top_k_slider"
    )

    # Sidebar for OpenAI LLM settings
    st.sidebar.header("OpenAI Settings", divider="red")

    def validate_openai_key(api_key):
        try:
            openai.api_key = api_key
            openai.Engine.list()
            logger.info("Valid OpenAI API key")
            return True
        except Exception as e:
            st.warning(e)
            logger.warning(f"Invalid OpenAI API key: {e}")
            return False

    openai_api_key = st.sidebar.text_input(
        label="#### Your OpenAI API key 👇",
        placeholder="Paste your openAI API key, sk-",
        type="password",
        key="openai_api_key",
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key

    model_name = st.sidebar.selectbox(
        "Select Language Model",
        ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct", "gpt-4"),
        index=0 if st.session_state.last_llm_model_name == "gpt-3.5-turbo-16k" else 1,
        key="model_name_select",
    )
    temperature = st.sidebar.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.temperature,
        0.01,
        key="temperature_slider",
    )

    top_n_for_llm = st.sidebar.slider(
        "Select top_n for LLM",
        1,
        new_top_k,  # Ensure the max value is always less than new_top_k
        st.session_state.top_n_for_llm,
        key="top_n_for_llm_slider",
    )

    def set_llm_prompt_text(llm_prompt_text):
        st.session_state.llm_prompt_text = llm_prompt_text
        # print(f"second Setting: {st.session_state.llm_prompt_text}")

    llm_prompt_text_placeholder = st.sidebar.empty()
    with llm_prompt_text_placeholder.container():
        llm_prompt_text = st.text_area(
            "LLM Prompt",
            value=st.session_state.llm_prompt_text,
            height=400,
            key=st.session_state.llm_prompt_text_area_key,
        )

    # Button to reload the default prompt
    if st.sidebar.button("Reload Default Prompt"):
        default_prompt = STUDYINFO_QA_PROMPT.get_template()
        st.session_state.llm_prompt_text_area_key += 1  # Increment key to force update
        llm_prompt_text_placeholder.empty()  # Empty the previous text_area
        with llm_prompt_text_placeholder.container():
            llm_prompt_text = st.text_area(
                "LLM Prompt",
                value=default_prompt,
                height=400,
                key=st.session_state.llm_prompt_text_area_key,
            )
            #   on_change=set_llm_prompt_text, args=(default_prompt,))
            set_llm_prompt_text(llm_prompt_text)

    # Initialize or update the engine with the API key and other settings
    engine = initialize_engine()

    with st.form("query_form", clear_on_submit=False):
        # User text input for the query
        user_query = str(
            st.text_input(
                "Please enter your query about studies in the Data Repository:"
            )
        )
        # Form submit button
        submitted = st.form_submit_button("Execute Query")
        if submitted:
            # Check if the query is empty
            if user_query.strip() == "":
                st.error("Please enter a query before executing.")
                st.stop()

            logger.info(f"Submitting Query: '{user_query}'")

            if st.session_state.query_mode != "Retrieval" and not openai_api_key:
                st.warning("Please enter an OpenAI API key to use full query mode.")
                st.stop()

            # Validate OpenAI API key if not in retriever_only mode
            if st.session_state.query_mode == "Retrieval-Augmented Generation (RAG)":
                if not validate_openai_key(openai_api_key):
                    st.stop()

            # Determine if an update is needed
            if st.session_state.query_mode == "Retrieval":
                update_needed = new_top_k != st.session_state.last_top_k_retriever
            else:  # RAG mode
                update_needed = any(
                    [
                        new_top_k != st.session_state.last_top_k_rag,
                        top_n_for_llm != st.session_state.last_top_n_for_llm,
                        llm_prompt_text != st.session_state.last_llm_prompt_text,
                        query_mode != st.session_state.query_mode,
                        model_name != st.session_state.last_llm_model_name,
                        temperature != st.session_state.temperature,
                    ]
                )

            if update_needed:
                logger.info("Engine update needed due to configuration changes")
                if st.session_state.query_mode == "Retrieval":
                    if new_top_k != st.session_state.top_k:
                        with st.status("Updating Retriever engine"):
                            engine.create_retriever_only_engine(
                                similarity_top_k=new_top_k,
                                rerank_top_n=new_top_k,
                                response_mode="no_text",
                                text_qa_template=None,
                            )
                else:
                    with st.status(
                        "Updating-Retrieval Augmented Generation (RAG) engine"
                    ):
                        engine.create_retriever_query_engine(
                            model_name=model_name,
                            temperature=temperature,
                            similarity_top_k=new_top_k,
                            rerank_top_n=new_top_k,
                            top_n_for_llm=top_n_for_llm,
                            text_qa_template=llm_prompt_text,
                        )

            # Update last used top_k and llm_prompt_text
            if st.session_state.query_mode == "Retrieval":
                logger.info(f"Updating Retrieval mode session state")
                st.session_state.last_top_k_retriever = new_top_k
            else:  # RAG mode
                logger.info(f"Updating RAG mode session state")
                st.session_state.last_top_k_rag = new_top_k
                st.session_state.last_top_n_for_llm = top_n_for_llm
                st.session_state.last_llm_prompt_text = llm_prompt_text
                st.session_state.last_llm_model_name = model_name
                st.session_state.temperature = temperature
            st.session_state.query_mode = query_mode

            # Perform query using the current engine
            if st.session_state.query_mode == "Retrieval":
                logger.info("Running Retriever engine")
                with st.status("Running Retriever engine"):
                    result = engine.retriever_engine.query(user_query)
                    result.response = 'No LLM response in "Retrieval" query mode.'
            else:
                logger.info("Running Retrieval Augmented Generation (RAG) engine")
                with st.status("Running Retrieval Augmented Generation (RAG) engine"):
                    result = engine.query_engine.query(user_query)
                    st.session_state.last_llm_prompt_text = (
                        llm_prompt_text  # Set the last llm_prompt_text
                    )
                    st.session_state.llm_prompt_text = llm_prompt_text

            # Process source_nodes to create a DataFrame
            logger.info("Processing source_nodes to create a DataFrame")
            source_nodes = result.source_nodes
            study_titles = [node.metadata.get("title", "N/A") for node in source_nodes]
            study_ids = [node.metadata.get("id", "N/A") for node in source_nodes]
            study_abstracts = [node.text for node in source_nodes]
            similarity_scores = [node.score for node in source_nodes]

            data = {
                "Study Title": study_titles,
                "Study ID": study_ids,
                "Cosine Similarity": similarity_scores,
                "Abstract": study_abstracts,
            }

            df = pd.DataFrame(data)
            df = df.sort_values(
                by=["Cosine Similarity"], ascending=False
            ).reset_index(drop=True)
            st.session_state.df = df

            # Display query
            st.subheader("Query")
            st.write(user_query)

            # Display LLM response
            st.subheader("LLM Response")
            logger.info(f"LLM Response received: {result.response}")
            st.write(result.response)

            # Display DataFrame of retrieved results
            st.subheader("Retrieved Studies")
            st.dataframe(df, use_container_width=True)

            # Append to history
            logger.info("Appending query and results to history")
            st.session_state.query_counter += 1
            st.session_state.history.append(
                {
                    "query": user_query,
                    "query_counter": st.session_state.query_counter,
                    "response": result.response,
                    "retrieved_studies": df.to_dict(),
                }
            )

    # # Create three columns. The first two columns will be for the download buttons,
    # # and the third column will be for the clear history button.
    emptycol, col2, col3, col4 = st.columns([2.5, 1, 1, 1])

    # Place the "Download Last Query in JSON" button in the first column
    with col2:
        if st.button("Download Last Query (JSON)", key="download_last_query_button"):
            if len(st.session_state.history) > 0:  # Ensure there's something in history
                last_query_results = st.session_state.history[-1]
                last_query_json = json.dumps(last_query_results, indent=4)
                b64 = base64.b64encode(
                    last_query_json.encode()
                ).decode()  # some bytes manipulation to encode as base64
                href = f'<a href="data:text/json;base64,{b64}" download="FITBIR-Data-Repository_query-results.json">Click to download last query results in JSON format</a>'
                st.markdown(href, unsafe_allow_html=True)

    # Place the "Download Entire Query History in JSON" button in the second column
    with col3:
        if st.button("Download Query History (JSON)", key="download_history_button"):
            history_json = json.dumps(st.session_state.history, indent=4)
            b64 = base64.b64encode(
                history_json.encode()
            ).decode()  # some bytes manipulation to encode as base64
            href = f'<a href="data:text/json;base64,{b64}" download="FITBIR-Data-Repository_query-history.json">Click to download your history in JSON format</a>'
            st.markdown(href, unsafe_allow_html=True)

    # Place the "Clear Query History" button in the third column and make it red
    with col4:
        if st.button(label=":red[Clear Query History]", key="clear_history_button"):
            st.session_state.history = []
            st.session_state.query_counter = 0
            st.success("Query History cleared!")


if __name__ == "__main__":
    main()
