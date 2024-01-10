import logging
import os
import sys
from typing import Any, Dict, Generator, List, Union

from openai import OpenAI
import openai
import streamlit as st
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ResponseType = Union[Generator[Any, None, None], Any, List, Dict]

openai.api_key = os.environ['OPENAI_API_KEY']

@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def load_index() -> Any:
    """Load the index from the storage directory."""
    print("Loading index...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_dir, "kb")

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    print("Done.")
    return query_engine

def if_empty(prompt):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    print("epmty")
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def main() -> None:
    """Run the chatbot."""
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = load_index()
        
    st.set_page_config(page_title="Chat with TaxGpt",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

    st.header("Chat with TaxGpt 🤖")
    st.write("Ask away your tax questions!")
    st.caption("\u00A9 Tahmid Hossain")

    if "messages" not in st.session_state:
        system_prompt = (
            "Please first try to answer the user's questions based on what you know about the document. "
            "if the documents dont have a good answer, use your knowledge but please reply something."
        )
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if message["role"] not in ["user", "assistant"]:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print("Querying query engine API...")
            response = st.session_state.query_engine.query(prompt)
            response = f"{response}"
            full_response = if_empty(response) if response == "Empty Response" else f"{response}"
            print(full_response)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
