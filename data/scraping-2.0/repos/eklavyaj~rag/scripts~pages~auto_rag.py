import datetime
import os
import time
import pandas as pd
import pickle as pkl
import transformers
import torch
import sqlite3
import streamlit as st
from dotenv import load_dotenv
import gc

from llama_index.llms import HuggingFaceLLM
from llama_index.llms import OpenAI

import scripts.utils.helper_unstructured_rag as unstructured_rag
import scripts.utils.helper_structured_rag as structured_rag
import scripts.utils.st_ingest as st_ingest

load_dotenv()

CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

DB_URL = os.getenv("DB_URL")
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_AUTO = os.getenv("EXPERIMENT_LOGGER_AUTO")
CHAT_HISTORY_AUTO = os.getenv("CHAT_HISTORY_AUTO")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")


@st.cache_resource
def get_llm(model_name, token, cache_dir):
    if model_name.lower() == "openai":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        return llm

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=cache_dir
    )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=token,
        trust_remote_code=True,
        cache_dir=cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=25,
        generate_kwargs={"temperature": 0.1},
        tokenizer=tokenizer,
        model_name=model_name,
        device_map="cuda:0",
        model_kwargs={
            "trust_remote_code": True,
            "config": model_config,
            "quantization_config": bnb_config,
            "use_auth_token": token,
            "cache_dir": cache_dir,
        },
    )

    return llm


def intent_classification(model_name, query):
    start = time.time()

    if model_name.lower() == "openai":
        template = f"""
        Structured data: user portfolio, portfolio, historical stock prices
        Unstructured data: news articles, qualitative questions, recent happenings
        
        Given the query {query}, identify which kind of data is required to answer this query. 
        Answer in one word and select from structured or unstructured.
        """
        llm = get_llm(model_name=model_name, token=TOKEN, cache_dir=CACHE_DIR)
        resp = llm.complete(template)

        text = resp.text.lower().replace(".", "").strip()

        if "unstructured" in text:
            text = "unstructured"
        else:
            text = "structured"

        return time.time() - start, text

    system_prompt = """
    Structured data: user portfolio, portfolio, historical stock prices
    Unstructured data: news articles, qualitative questions, recent happenings
    
    Identify which kind of data is required to answer the given query. 
    Answer in one word and select from structured or unstructured.
    """

    template = f"""<s>[INST] <<SYS>>
        { system_prompt }
        <</SYS>>

        { query } [/INST]
    """

    llm = get_llm(model_name=model_name, token=TOKEN, cache_dir=CACHE_DIR)
    resp = llm.complete(template)
    text = resp.text.lower().replace(".", "").strip()

    if "unstructured" in text:
        text = "unstructured"

    else:
        text = "structured"

    return time.time() - start, text


def unstructured_answer_query(query_engine, query):
    start = time.time()
    response = query_engine.query(query)
    end = time.time()

    return end - start, response


def structured_answer_query(query_engine, query):
    time, resp, sql = structured_rag.get_response(
        query_engine=query_engine, query_str=query, print_=False
    )
    return time, resp, sql


def clean_response(response):
    response = response.replace("$", "\$")
    response = response.replace('"', "'")

    return response


def render(history_file, models, model_names_to_id, portfolios):
    st.sidebar.divider()

    model = st.sidebar.selectbox("Choose Model", models)
    if model_names_to_id[model].lower() == "openai":
        openai_key = st.sidebar.text_input("OpenAI API Key")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        else:
            st.info("Please Enter OpenAI's API Key to continue.")
            st.stop()

    elif model == "Enter HuggingFace ID":
        model_id = st.sidebar.text_input("Enter HuggingFace Model ID")
        if model_id:
            model_names_to_id[model_id] = model_id
            model = model_id
        else:
            st.info("Please Enter HuggingFace Model ID to continue.")
            st.stop()

    portfolio = st.sidebar.selectbox("Choose Portfolio", portfolios)

    with st.spinner("Loading Unstructured RAG"):
        unstructured_service_context = unstructured_rag.get_service_context(
            model_names_to_id[model], token=TOKEN, cache_dir=CACHE_DIR
        )

        try:
            unstructured_query_engine = unstructured_rag.get_query_engine(
                model_name=model_names_to_id[model],
                service_context=unstructured_service_context,
            )
        except:
            unstructured_rag.load_docs_and_save_index(
                model_names_to_id[model], service_context=unstructured_service_context
            )
            unstructured_query_engine = unstructured_rag.get_query_engine(
                model_name=model_names_to_id[model],
                service_context=unstructured_service_context,
            )

    with st.spinner("Loading Structured RAG"):
        sql_database = structured_rag.get_database(
            DB_URL,
            EXCEL_FILE_PATH=EXCEL_FILE_PATH,
            portfolio=portfolio,
            SOURCE_DOCUMENTS_PATH=SOURCE_DOCUMENTS_PATH,
            csv_path=ASSET_MAPPING_PATH,
        )

        structured_service_context = structured_rag.get_service_context(
            model_names_to_id[model], token=TOKEN, cache_dir=CACHE_DIR
        )

        structured_query_engine = structured_rag.get_query_engine(
            sql_database=sql_database, service_context=structured_service_context
        )

    st.sidebar.divider()
    refresh_db = st.sidebar.button(
        "Refresh News", use_container_width=True, help="Might take a while to complete"
    )

    if refresh_db:
        try:
            st_ingest.st_ingest_data()
            unstructured_rag.load_docs_and_save_index(
                model_names_to_id[model], service_context=unstructured_service_context
            )
        except:
            st.sidebar.write("Failed to Refresh DB")

    clear_history = st.sidebar.button("Clear History", use_container_width=True)
    if clear_history:
        if os.path.exists(history_file):
            os.remove(history_file)

    try:
        st.session_state.messages = pkl.load(open(history_file, "rb"))
    except:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(clean_response(message["content"]))

        else:
            with st.chat_message(message["role"]):
                if message["type"] == "structured":
                    st.markdown(clean_response(message["content"]) + "\n")

                    with st.expander("SQL", expanded=False):
                        st.code(message["sql"], language="sql")
                        if type(message["df_sql"]) == str:
                            st.write(message["df_sql"])
                        else:
                            st.dataframe(message["df_sql"], hide_index=True)

                else:
                    st.markdown(clean_response(message["content"]) + "\n")

                    with st.expander("Sources"):
                        tabs = st.tabs(
                            [f"Source {i+1}" for i in range(len(message["sources"]))]
                        )
                        for i, source in enumerate(message["sources"]):
                            tabs[i].link_button(source["title"], source["url"])
                            tabs[i].markdown(source["date"])
                            tabs[i].markdown(clean_response(source["text"]))

                            if i == 3:
                                break

            with st.chat_message("⚒️"):
                st.write(f"Time Taken: {message['time']:0.3f} seconds")
                st.write(f"Model: {message['model']}")
                st.write(f"Intent: {message['type'].capitalize()}")
                st.write(f"Portfolio: {message['portfolio']}")

    if input_query := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)

        with st.spinner("Getting Response"):
            with st.spinner("Classifying Intent"):
                time_intent, intent = intent_classification(
                    model_names_to_id[model], input_query
                )

            with st.spinner(f"Activating {intent.capitalize()} Engine"):
                if intent == "unstructured":
                    time, resp = unstructured_answer_query(
                        query_engine=unstructured_query_engine, query=input_query
                    )
                    sql = None
                else:
                    time, resp, sql = structured_answer_query(
                        query_engine=structured_query_engine, query=input_query
                    )

        sources = []
        df_sql = None

        if intent == "unstructured":
            source_nodes = resp.source_nodes
            resp = resp.response
            for node in source_nodes:
                title = node.metadata["title"].replace("_", " ")
                date = title.rsplit(" ", 1)[-1].split(".")[0]
                title = title.rsplit(" ", 1)[0]

                temp = {}
                temp["url"] = ":".join(node.text.split("\n")[0].split(":")[1:])
                temp["title"] = title
                temp["date"] = f"*Dated: {date}*"
                temp["text"] = "\n".join(node.text.rsplit("\n", -2)[2:])

                sources.append(temp)
        else:
            try:
                conn = sqlite3.connect(DB_URL)
                df_sql = pd.read_sql(sql, con=conn)
                conn.close()
            except:
                df_sql = "Incorrect Query!"

        with st.chat_message("assistant"):
            st.markdown(clean_response(resp))

            if intent == "unstructured":
                with st.expander("Sources"):
                    tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
                    for i, source in enumerate(sources):
                        tabs[i].link_button(source["title"], source["url"])
                        tabs[i].markdown(source["date"])
                        tabs[i].markdown(clean_response(source["text"]))

                        if i == 3:
                            break
            else:
                with st.expander("SQL", expanded=True):
                    st.code(sql, language="sql")
                    if type(df_sql) == str:
                        st.write(df_sql)
                    else:
                        st.dataframe(df_sql, hide_index=True)

        with st.chat_message("⚒️"):
            st.write(f"Time Taken: {time + time_intent:0.3f} seconds")
            st.write(f"Model: {model}")
            st.write(f"Intent: {intent.capitalize()}")
            st.write(f"Portfolio: {portfolio}")

        timestamp = datetime.datetime.now()

        message = {
            "role": "assistant",
            "timestamp": timestamp,
            "portfolio": portfolio,
            "model": model,
            "type": intent,
            "content": resp,
            "time": time + time_intent,
            "sources": sources,
            "sql": sql,
            "df_sql": df_sql,
        }

        st.session_state.messages.append(message)

        df = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "model": [model],
                "portfolio": [portfolio],
                "user_input": [input_query],
                "intent": [intent],
                "llm_response": [resp],
                "sql_query": [sql],
                "sources": [sources],
                "time_taken": [time + time_intent],
            }
        )

        try:
            results = pd.read_csv(EXPERIMENT_LOGGER_AUTO)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER_AUTO, index=False)

        except:
            df.to_csv(EXPERIMENT_LOGGER_AUTO, index=False)

        try:
            pkl.dump(st.session_state.messages, open(history_file, "wb"))
        except:
            pass
