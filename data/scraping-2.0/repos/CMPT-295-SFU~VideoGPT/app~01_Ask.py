import pandas as pd
from streamlit_text_rating.st_text_rater import st_text_rater
from streamlit_star_rating import st_star_rating
import sys
from loguru import logger
import time
import base64
import os
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from components.sidebar import sidebar
from utils import (
    query_gpt,
    query_gpt_memory,
    show_pdf,
    pinecone_api_key,
    openai_api_key,
)
import pinecone
from openai import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text = ""


def display_slide_results(slide_results):
    for m in slide_results["results"][0]["matches"][0:2]:
        url = m["metadata"]["file"]
        page = m["metadata"]["Slide"]
        st.markdown(
            f"-[{url}](https://www.cs.sfu.ca/~ashriram/Courses/CS295//assets/lectures/{url}#page={page})"
        )


def extract_week(slide_results):
    week = (
        slide_results["results"][0]["matches"][0]["metadata"]["file"]
        .split("/")[0]
        .replace("Part", "")
    )
    return week

def display_audio_results(audio_results):
    for m in audio_results["results"][0]["matches"][0:2]:
        url = m["metadata"]["url"]
        st.markdown(f"-[{url}]({url})")


st.set_page_config(
    page_title="295VideoGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "295VideoGPT is a semantic bot that answers questions about your video lectures",
    },
)

if "count" not in st.session_state:
    st.session_state.count = 0


# Add a rotating log file that rotates every day
# logger.add("logs/myapp_{time:YYYY-MM-DD}.log", rotation="1 week")

client = OpenAI(api_key=openai_api_key)


st.header("295Bot")
col1, col2 = st.columns(2)


with col1:
    st.subheader("Top 3 Video Topic Search")
    #    st.subheader("Lookup a topic")
    st.markdown(
        """
        Here are some examples
        - `R-type instruction`, `Memory allocation`
        """
    )
    query = st.text_area("Enter topic you want to find in videos", max_chars=25)

    if st.button("Ask"):
        current = st.session_state.count
        st.session_state.count += 1
        if query == "":
            st.error("Please enter a topic")
        with st.spinner("Generating answer..."):
            # res = query_gpt_memory(chosen_class, chosen_pdf, query)
            index_id = "295-youtube-index"
            pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
            index = pinecone.Index(index_id)
            # encoded_query = client.embeddings.create(input=query,   model="text-embedding-ada-002")['data'][0]['embedding']
            # res = query_gpt(chosen_class, chosen_pdf, query)
            text = query.replace("\n", " ")
            encoded_query = (
                client.embeddings.create(input=[text], model="text-embedding-ada-002")
                .data[0]
                .embedding
            )
            slide_results = index.query(
                queries=[encoded_query],
                top_k=5,
                namespace="Slides",
                include_metadata=True,
            )
            audio_results = index.query(
                queries=[encoded_query], top_k=5, include_metadata=True
            )
            st.markdown("**Slide References**")
            display_slide_results(slide_results)
            week = extract_week(slide_results)
            st.markdown(
                f"[Relevant Week's videos](https://www.cs.sfu.ca/~ashriram/Courses/CS295/videos.html#week{week})\n`you can lookup using provided slide reference above`"
            )
            st.markdown("**Video References**")
            st.markdown(f"`May be off a bit since it uses instructor's audio`")
            display_audio_results(audio_results)
            
            st.session_state.topic = f"Topic: {query} | "
            log_query = query.replace("\n", " ")
            logger.bind(user=st.session_state.count).info(f"Topic: {log_query} |")

topic_rating = st_text_rater(text="Was it helpful?", key="topic_text")


with st.container():
    st.subheader("Question-Answering: I am feeling Lucky")
    #    st.subheader("Lookup a topic")
    st.markdown(
        """
        Here are some examples
        - `What is a pointer and give me an example ?`
        - `How to find the address of an element in a 2D array ?` 
        """
    )
    query = st.text_area("Question", max_chars=200)
    model_options = ["gpt-4-1106-preview"]
    option = st.selectbox(
        "Select chat model  **gpt-4 models are more accurate but could take longer**",
        set(model_options),
    )
    if st.button("Answer"):
        st.session_state.count += 1
        if query == "":
            st.error("Please enter a topic")
        else:
            # concert to set
            delay = 20
            if option == "gpt-4-1106-preview":
                delay = 40
            with st.spinner(f"Generating answer.....(can take upto {delay} seconds)"):
                index_id = "295-youtube-index"
                pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
                index = pinecone.Index(index_id)
                index.describe_index_stats()

                # Encode the query using the 'text-embedding-ada-002' model
                # encoded_query = client.embeddings.create(input=query,model="text-embedding-ada-002")['data'][0]['embedding']
                encoded_query = (
                    client.embeddings.create(
                        input=[query], model="text-embedding-ada-002"
                    )
                    .data[0]
                    .embedding
                )
                context = "Question: " + query + "\n"
                context += "\n" + "#######Slide Context#####\n"
                slide_results = index.query(
                    queries=[encoded_query],
                    top_k=5,
                    namespace="Slides",
                    include_metadata=True,
                )
                for r in slide_results["results"][0]["matches"]:
                    context += r["metadata"]["description"] + "\n"

                context += "\n #######Audio Context#####\n"
                audio_results = index.query(
                    queries=[encoded_query], top_k=5, include_metadata=True
                )

                for r in audio_results["results"][0]["matches"]:
                    context += r["metadata"]["text"] + "\n"
                query_gpt = [
                    {
                        "role": "system",
                        "content": "You are an expert in RISC-V and Computer Architecture. Try to answer the following questions based on the context below in markdown format. If you don't know the answer, just say 'I don't know'.",
                    },
                    {"role": "user", "content": f"""{context}"""},
                ]
                chat_box = st.empty()
                display_handler = StreamDisplayHandler(chat_box, display_method="write")
                chat = ChatOpenAI(
                    model=option,
                    max_tokens=4096,
                    streaming=True,
                    callbacks=[display_handler],
                    openai_api_key=openai_api_key,
                )
                answer_response = chat(
                    [
                        SystemMessage(
                            content="You are an expert in RISC-V and Computer Architecture. Try to answer the following questions based on the context below in markdown format. If you don't know the answer, just say 'I don't know'."
                        ),
                        HumanMessage(content=f"""{context}"""),
                    ]
                )

                st.markdown("**Slide References**")
                display_slide_results(slide_results)
                week = extract_week(slide_results)
                st.markdown(
                    f"[Relevant Week's videos](https://www.cs.sfu.ca/~ashriram/Courses/CS295/videos.html#week{week})\n`you can lookup using provided slide reference above`"
                )
                st.markdown("**Video References**")
                st.markdown(f"`May be off a bit since it uses instructor's audio`")
                log_query = query.replace("\n", " ")
                logger.bind(user=st.session_state.count).info(f"Q/A: {log_query}|")
                st.session_state.query = f"Q/A: {query}"

query_rating = st_text_rater(text="Was it helpful?", key="query_text")

if query_rating is not None:
    if "query" in st.session_state:
        log_query = st.session_state.query.replace("\n", " ")
        logger.bind(user=st.session_state.count).info(
            log_query + f" | Rating: {query_rating}"
        )


if topic_rating is not None:
    if "topic" in st.session_state:
        log_topic = st.session_state.topic.replace("\n", " ")
        logger.bind(user=st.session_state.count).info(
            log_topic + f" | Rating: {topic_rating}"
        )
