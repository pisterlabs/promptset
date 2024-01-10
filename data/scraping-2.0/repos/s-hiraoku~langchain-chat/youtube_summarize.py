import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import ssl
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chain_type_select():
    st.sidebar.markdown("## Chain Type")
    chain_type = st.sidebar.radio(
        "Choose a chain type:",
        [
            "stuff",
            "map_reduce",
        ],
    )
    return chain_type


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url


def get_document(url):
    with st.spinner("Fetching Content ..."):
        ssl._create_default_https_context = ssl._create_unverified_context
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size=st.session_state.max_token,
            chunk_overlap=0,
        )
        return loader.load_and_split(text_splitter=text_splitter)


def summarize(llm, docs, chain_type):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.

============

{text}

============

ここから日本語で書いてね
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    common_args = {"llm": llm, "chain_type": chain_type, "verbose": True}

    with get_openai_callback() as cb:
        if chain_type == "stuff":
            specific_args = {"prompt": PROMPT}
        elif chain_type == "map_reduce":
            specific_args = {"map_prompt": PROMPT, "combine_prompt": PROMPT}

        chain = load_summarize_chain(**common_args, **specific_args)
        response = chain(
            {"input_documents": docs, "token_max": st.session_state.max_token},
            return_only_outputs=True,
        )

    return response["output_text"], cb.total_cost


def handle_youtube_summarize(llm, chain_type):
    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            try:
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document, chain_type)
                    st.session_state.costs.append(cost)
            except Exception as e:
                st.error(f"ドキュメントが長い場合は、Chan Type を map_reduce にしてください。\n\n${e}")
                output_text = None
        else:
            output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
