import os
import argparse
import openai
import streamlit as st

from prompts import SYSTEM_PROMPT, construct_user_prompt
from knowledge_base import VectorRetrieval

if os.path.exists('.streamlit/secrets.toml') or os.path.exists('~/.streamlit/secrets.toml'):
    if "OPENAI_API_KEY" in st.secrets:
        # Read OPENAI_API_KEY from `.streamlit/secrets.toml`
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        print('Loaded OPENAI_API_KEY from secrets.toml')

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', help='openai model, default gpt-3.5-turbo')
parser.add_argument('--max_tokens', required=False, type=int, default=1024, help='max_tokens, default 1024')
parser.add_argument('--temperature', required=False, type=float, default=0, help='temperature, default 0')
parser.add_argument('--max_history_len', required=False, type=int, default=5, help='max history length, default 5')
parser.add_argument('--knowledge_dir', required=False, type=str, default='./knowledge/docs', help='directory of knowledge files, default ./knowledge/docs')
args = parser.parse_args()

def main_web():
    st.title("Chat With ChatGPT")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = args.model

    if "vector_retrieval" not in st.session_state:
        if args.knowledge_dir is not None and os.path.exists(args.knowledge_dir):
            st.session_state["vector_retrieval"] = None
            print('Specified knowledge path not exist.')
        else:
            st.session_state["vector_retrieval"] = VectorRetrieval('./knowledge/vector_db/')
            n_texts = st.session_state["vector_retrieval"].add_index_for_docs(path=args.knowledge_dir)
            if n_texts == 0:
                st.session_state["vector_retrieval"] = None
                print('No knowledge files specified.')
            else:
                print('Generate knowledge base success.')

    if "messages" not in st.session_state:
        st.session_state['messages'] = []
        print(st.session_state.keys())
        st.session_state['messages'].append({"role": "system", "content": SYSTEM_PROMPT})

    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {
                        "role": m["role"], 
                        "content": construct_user_prompt(m["content"], st.session_state["vector_retrieval"])
                    } for m in st.session_state['messages'][-args.max_history_len:]
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state['messages'].append({"role": "assistant", "content": full_response})

main_web()