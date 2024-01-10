import openai
import streamlit as st
import os, sys
from omegaconf import OmegaConf
import argparse

from chroma_retriever import ChromaRetriever
from condense_questions import condense_questions
from templates import system_provider_template, system_patient_template

def parse_args(args):
    parser = argparse.ArgumentParser(description='demo how to use streamlit for ai embeddings to question/answer.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    parser.add_argument("-r", "--role", dest="role",
                        help="role(patient/provider) for question/answering", metavar="ROLE")
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])
if args.yamlfile is None:
    os._exit(-1)

yamlfile = args.yamlfile
config = OmegaConf.load(yamlfile)

role = args.role

chroma_retriever = ChromaRetriever(config)

st.title('PGx SLCOLB1 Chatbot')
st.subheader("Help us help you learn about PGx SLCOLB1")
st.session_state["openai_model"] = config.openai.chat_model_name

openai.api_key = os.getenv("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"What do you want to know: ", key="input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]

    standalone_question = condense_questions(questions, prompt, model=st.session_state["openai_model"])

    docs = chroma_retriever.max_marginal_relevance_search(standalone_question)
    context = ""
    for doc in docs:
        context = context + f"; content: { doc.page_content}" + f". source: {doc.metadata['source']} "

    if role == "provider":
        system_template = system_provider_template
    elif role == "patient":
        system_template = system_patient_template
    else:
        print("role not supported")
        exit()

    system_template_content = system_template.format(context=context)

    messages.append({"role": "system", "content": system_template_content})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
