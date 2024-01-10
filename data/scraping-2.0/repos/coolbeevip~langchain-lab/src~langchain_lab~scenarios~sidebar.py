# Copyright 2023 Lei Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import langchain
import streamlit as st

from langchain_lab import logger
from langchain_lab.core.translate import LANGUAGES
from src.langchain_lab.core.embedding import embedding_init
from src.langchain_lab.core.huggingface import download_hugging_face_model
from src.langchain_lab.core.llm import TrackerCallbackHandler, llm_init, load_llm_chat_models

AI_PLATFORM = {
    "OpenAI": {
        "api_url": os.environ.get("OPENAI_API_BASE", ""),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "embedding": {
            "provider": "openai",
            "model": ["text-embedding-ada-002"],
            "model_kwargs": {},
        },
    },
    "FastChat": {
        "api_url": os.environ.get("FASTCHAT_API_BASE", ""),
        "api_key": os.environ.get("FASTCHAT_API_KEY", ""),
        "embedding": {
            "provider": "huggingface",
            "model": [
                "moka-ai/m3e-base",
                "sentence-transformers/msmarco-distilbert-base-v4",
                "shibing624/text2vec-base-chinese",
            ],
            "model_kwargs": {"device": "cpu"},
        },
    },
    "LMStudio": {
        "api_url": os.environ.get("LMSTUDIO_API_BASE", ""),
        "api_key": os.environ.get("LMSTUDIO_API_KEY", ""),
        "embedding": {
            "provider": "huggingface",
            "model": [
                "moka-ai/m3e-base",
                "sentence-transformers/msmarco-distilbert-base-v4",
                "shibing624/text2vec-base-chinese",
            ],
            "model_kwargs": {"device": "cpu"},
        },
    },
    "BaiChuan": {
        "api_url": os.environ.get("BAICHUAN_API_BASE", ""),
        "api_key": os.environ.get("BAICHUAN_API_KEY", ""),
        "embedding": {
            "provider": "huggingface",
            "model": [
                "moka-ai/m3e-base",
                "sentence-transformers/msmarco-distilbert-base-v4",
                "shibing624/text2vec-base-chinese",
            ],
            "model_kwargs": {"device": "cpu"},
        },
    },
}


@st.cache_resource
def load_embedding(embedding_provider, model_kwargs):
    logger.info(f"Loading embedding provider {embedding_provider}")
    if embedding_provider == "huggingface":
        with st.spinner(f"Loading model {st.session_state['EMBED_MODEL_NAME']}...⏳"):
            try:
                download_hugging_face_model(st.session_state["EMBED_MODEL_NAME"])
            except Exception as e:
                st.error(e)
    embedding_init(
        provider=embedding_provider,
        api_url=st.session_state["API_URL"],
        api_key=st.session_state["API_KEY"],
        model_name=st.session_state["EMBED_MODEL_NAME"],
        model_kwargs=model_kwargs,
    )


def left_sidebar():
    if "DEBUG_CALLBACK" not in st.session_state:
        st.session_state["DEBUG_CALLBACK"] = TrackerCallbackHandler(st)
    else:
        st.session_state["DEBUG_CALLBACK"].clean_tracks()

    with st.sidebar:
        st.markdown("# LangChain Lab")
        platforms = [k.strip() for k in os.environ["DEFAULT_AI_PLATFORM_SUPPORT"].split(",")]
        enabled_platforms = [k for k in AI_PLATFORM.keys() if k in platforms]
        ai_platform = st.selectbox(
            "PLATFORM",
            list(enabled_platforms),
            label_visibility="hidden",
            index=list(enabled_platforms).index(os.environ.get("DEFAULT_AI_PLATFORM", "OpenAI")),  # noqa: E501
        )
        st.session_state["AI_PLATFORM"] = ai_platform

        with st.expander("MODEL SETTINGS", expanded=True):
            # API Settings
            api_url_input = st.text_input(
                "URL",
                type="password",
                placeholder="Paste your AI Platform API URL here (https://-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
                value=AI_PLATFORM[ai_platform]["api_url"],
            )
            api_key_input = st.text_input(
                "KEY",
                type="password",
                placeholder="Paste your AI Platform key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
                value=AI_PLATFORM[ai_platform]["api_key"],
            )
            st.session_state["API_URL"] = api_url_input
            st.session_state["API_KEY"] = api_key_input

            model_names = load_llm_chat_models(api_url_input, api_key_input)

            # LLM Settings
            api_model_name = st.selectbox("CHAT MODEL", model_names)
            api_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            st.session_state["API_MODEL_NAME"] = api_model_name
            st.session_state["API_TEMPERATURE"] = api_temperature

            # Stream API
            stream_api = st.toggle("STREAM API", value=True)
            st.session_state["STREAM_API"] = stream_api

            # Initialize LLM
            llm = llm_init(api_url_input, api_key_input, api_model_name, api_temperature, stream_api)
            if llm is None:
                del st.session_state["LLM"]
            else:
                st.session_state["LLM"] = llm

            if "LLM" in st.session_state:
                st.info(f"Initialized LLM with model_name={llm.model_name}, temperature={llm.temperature}")
            else:
                st.button("REFRESH")

        # Debug Settings
        api_debug = st.toggle("DEBUG")
        st.session_state["LANGCHAIN_DEBUG"] = api_debug
        langchain.debug = st.session_state["LANGCHAIN_DEBUG"]

        scenario = st.selectbox("SCENARIO", ["CHAT", "DOCUMENT"])
        st.session_state["SCENARIO"] = scenario
        if scenario == "CHAT":
            # Memory Settings
            st.session_state["CHAT_MEMORY_ENABLED"] = st.toggle("MEMORY", True)
            if st.session_state["CHAT_MEMORY_ENABLED"]:
                with st.expander("MEMORY", expanded=True):
                    if st.session_state["CHAT_MEMORY_ENABLED"]:
                        chat_memory_history_deep = st.slider("HISTORY DEEP", 20, 100, 20)
                        st.session_state["CHAT_MEMORY_HISTORY_DEEP"] = chat_memory_history_deep
                        if st.button("NEW SESSION"):
                            st.session_state.chat_messages = []

            with st.expander("ROLE", expanded=True):
                st.text_area("SYSTEM PROMPT", key="SYSTEM_PROMPT", height=200,
                             placeholder="Enter a prompt word related to the role")
                if st.button("CONFIRM"):
                    st.session_state.chat_messages = []
                    st.session_state["CHAT_PROMPT_TEMPLATE"] = st.session_state.get("SYSTEM_PROMPT", "")
                    st.success("System prompt confirmed")

                st.info("chat history placeholder is {chat_history}")

        elif scenario == "DOCUMENT":
            with st.expander("CHAIN", expanded=True):
                # Chain Settings
                chain_type = st.selectbox(
                    "CHAIN TYPE",
                    ("stuff", "map_reduce", "refine", "map_rerank"),
                )
                st.session_state["CHAIN_TYPE"] = chain_type
                if chain_type == "stuff":
                    st.info(
                        "The stuff documents chain ('stuff' as in 'to stuff' or 'to fill') \
                    is the most straightforward of the document chains. It takes a list of documents, \
                    inserts them all into a prompt and passes that prompt to an LLM. \
                    The LLM then generates a response based on all the documents."
                    )
                elif chain_type == "refine":
                    st.info(
                        "The refine documents chain constructs a response by looping over the input documents and \
                    iteratively updating its answer. For each document, it passes all non-document inputs, the current document,\
                     and the latest intermediate answer to an LLM chain to get a new answer"
                    )
                elif chain_type == "map_reduce":
                    st.info(
                        "The map reduce documents chain first applies an LLM chain to each document \
                    individually (the Map step), treating the chain output as a new document. \
                    It then passes all the new documents to a separate combine documents chain to \
                    get a single output (the Reduce step). It can optionally first compress, or collapse, \
                    the mapped documents to make sure that they fit in the combine \
                    documents chain (which will often pass them to an LLM). \
                    This compression step is performed recursively if necessary."
                    )
                elif chain_type == "map_rerank":
                    st.info(
                        "The map re-rank documents chain runs an initial prompt on each document, \
                    that not only tries to complete a task but also gives a score for how certain it is in its answer. \
                    The highest scoring response is returned."
                    )
                else:
                    st.error("Chain type not supported")

                # Summary Settings
                summary_language = st.selectbox("TRANSLATE", LANGUAGES.keys())
                st.session_state["SUMMARY_LANGUAGE"] = summary_language
                st.info(f"Answer in {summary_language}")

            with st.expander("TEXT SPLITTERS", expanded=True):
                # Embedding Settings
                embedding_model = st.selectbox("EMBEDDING MODEL", AI_PLATFORM[ai_platform]["embedding"]["model"])
                st.session_state["EMBED_MODEL_NAME"] = embedding_model

                load_embedding(
                    AI_PLATFORM[ai_platform]["embedding"]["provider"],
                    AI_PLATFORM[ai_platform]["embedding"]["model_kwargs"],
                )
                chunk_size = st.slider("Chunk Size", 0, 5000, 200)
                chunk_overlap = st.slider("Chunk Overlap", 0, chunk_size, 20)
                st.session_state["CHUNK_SIZE"] = chunk_size
                st.session_state["CHUNK_OVERLAP"] = chunk_overlap

                embed_top_k = st.slider("Top K", 0, 50, 3)
                st.session_state["EMBED_TOP_K"] = embed_top_k
