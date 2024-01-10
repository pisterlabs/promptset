import os
import logging
import json
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st
import openai
from opencensus.ext.azure.log_exporter import AzureLogHandler

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
APPLICATIONINSIGHTS_CONNECTION_STRING = os.environ[
    "APPLICATIONINSIGHTS_CONNECTION_STRING"
]
CHAT_MODEL = "chat"  # gpt-35-turbo
EMBEDDING_MODEL = "text-embedding-ada-002"  # text-embedding-ada-002

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = AzureLogHandler(connection_string=APPLICATIONINSIGHTS_CONNECTION_STRING)
logger.addHandler(handler)
openai.api_type = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = "2023-07-01-preview"

st.set_page_config(page_title="Prompt monitoring playground")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def record_feedback(completion_id, feedback):
    # We make assumptions here that the conversation is
    # always user/ai/user/ai.  Once we have other steps
    # for RAG, we'll do a little more bookkeeping to find
    # the most recent prompt from the user, ignoring any
    # intermediate completions.
    conversation = []
    messages = st.session_state["chat_history"]

    for i in range(0, len(messages)):
        message = messages[i]
        conversation.append({"role": message["role"], "content": message["content"]})

        if message["completion"] and message["completion"].id == completion_id:
            message["feedback"] = feedback
            logger.info(
                "feedback",
                extra={
                    "custom_dimensions": {
                        "completionId": message["completion"].id,
                        "feedback": feedback,
                    }
                },
            )
            return


def show_chat_history():
    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]
        feedback = message["feedback"]
        completion = message["completion"]

        if completion:
            if message["feedback"]:
                col1, col2 = st.columns([10, 1])
                with col1:
                    with st.chat_message(name=role):
                        st.write(content)
                with col2:
                    if feedback == "good":
                        st.write("👍")
                    elif feedback == "bad":
                        st.write("👎")
            else:
                col1, col2, col3 = st.columns([10, 1, 1])
                with col1:
                    with st.chat_message(name=role):
                        st.write(content)
                with col2:
                    st.button(
                        "👍",
                        on_click=record_feedback,
                        args=(completion.id, "good"),
                        key=f"{completion.id}-good",
                    )
                with col3:
                    st.button(
                        "👎",
                        on_click=record_feedback,
                        args=(completion.id, "bad"),
                        key=f"{completion.id}-bad",
                    )
        else:
            with st.chat_message(name=role):
                st.write(content)

        if completion:
            st.json(completion, expanded=False)


def append_to_chat_history(role, content, completion=None):
    st.session_state["chat_history"].append(
        {
            "role": role,
            "content": content,
            "feedback": None,
            "completion": completion,
        }
    )


def get_conversation():
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat_history"]
    ]


prompt = st.chat_input("Ask a question")

if prompt:
    append_to_chat_history("user", prompt)

    start = datetime.now()
    completion = openai.ChatCompletion.create(
        engine=CHAT_MODEL, messages=get_conversation()
    )
    finish = datetime.now()
    elapsed = round((finish - start).total_seconds() * 1000)
    response = completion.choices[0].message.content
    role = completion.choices[0].message.role

    append_to_chat_history(role, response, completion)
    conversation = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat_history"]
    ]
    # The annotations are not always there, not sure why, handle it.
    prompt_filter_results = (
        completion.prompt_annotations[0].content_filter_results
        if hasattr(completion, "prompt_annotations") and completion.prompt_annotations
        else {}
    )
    response_filter_results = (
        completion.choices[0].content_filter_results
        if hasattr(completion.choices[0], "content_filter_results")
        and completion.choices[0].content_filter_results
        else {}
    )
    logger.info(
        "chat completion",
        extra={
            "custom_dimensions": {
                "completionId": completion.id,
                "prompt": prompt,
                "response": response,
                "conversation": json.dumps(conversation),
                "usage": json.dumps(completion.usage),
                "promptFilterResults": json.dumps(prompt_filter_results),
                "responseFilterResults": json.dumps(response_filter_results),
                "timeInMilliseconds": elapsed,
            }
        },
    )

    handler.flush()

show_chat_history()
