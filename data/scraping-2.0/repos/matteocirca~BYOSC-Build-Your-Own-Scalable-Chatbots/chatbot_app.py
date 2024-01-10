import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS

from backend.core import run_llm, get_faiss_vectordb

from openai import OpenAI


# ChatGPT integration
# Add an option to choose between Llama and ChatGPT
model_choice = st.sidebar.radio("Select Model:", ["Llama", "ChatGPT"])

if model_choice == "ChatGPT":
    st.session_state.OPENAI_API_KEY = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )

# check if openai api key is set
if model_choice == "ChatGPT" and 'OPENAI_API_KEY' not in st.session_state:
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# check if openai api key is set correctly
if model_choice == "ChatGPT" and 'OPENAI_API_KEY' in st.session_state and not st.session_state.OPENAI_API_KEY.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# TODO: Regenerate response if user edits prompt or unsatisfied with response


##############################################################################################
# Part of the code refactored from https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/ #
##############################################################################################

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "I'm your course buddy. How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "I'm your course buddy. How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Format for Llama2 chatbot following the LLaMA2 template
# <s>[INST] <<SYS>>
# System prompt
# <</SYS>>
#
# User prompt [/INST] Model answer </s>
def chat_completion_llama2(prompt):
    string_dialogue = "<s>[INST] <<SYS>>\n"
    string_dialogue += prompt
    string_dialogue += "\n<</SYS>>\n\n"

    # skip the first message since it is the system prompt
    for dict_message, i in zip(st.session_state.messages[1:], range(len(st.session_state.messages[1:]))):
        if dict_message["role"] == "user":
            string_dialogue += dict_message["content"] if i == 0 else "[INST] " + dict_message["content"]
            string_dialogue += " [/INST] "
        else:
            string_dialogue += dict_message["content"] + " "
            # string_dialogue += dict_message["content"] + " </s><s>"

    return string_dialogue

# Format for ChatGPT
# \nUser: User prompt\nAssistant: Model answer
def chat_completion_chatgpt(prompt):
    messages = []
    string_dialogue = ""

    messages.append({"role": "system", "content": prompt})

    # skip the first message since it is the system prompt
    for dict_message, i in zip(st.session_state.messages[1:], range(len(st.session_state.messages[1:]))):
        if dict_message["role"] == "user":
            string_dialogue += "\nUser: "
        else:
            string_dialogue += "\nAssistant: "
        string_dialogue += dict_message["content"]

    messages.append({"role": "user", "content": string_dialogue})

    return messages

# Function for generating RAG response
def run_rag_llama2(prompt_input, search):
    with st.spinner("Retrieving results..."):

        db = get_faiss_vectordb(inference_api_key=os.getenv('INFERENCE_API_KEY'))

        # load the FAISS vector database from the generated index path
        # retrieve the top 5 most similar documents
        docs_and_scores = db.similarity_search_with_score(search)[:5]

        prompt = "<s>[INST] <<SYS>>\nAnswer a question using References. Remember to mention the source(s) you use. For example, if 'Source: 01_introduction.pdf, Page: 8', you can say 'According to page 8 in slides 01_introduction.pdf, ...'. If no relevant answer is found, you can say 'My apologies, I didn't find any answer to your question. Please ask another question or rephrase your question.'\n<</SYS>>\n\n"
        prompt += f"Question: {prompt_input} References: "
        for doc, score in docs_and_scores:
            prompt += f"{doc} "
        prompt += "[/INST] "

        # TODO: Why LLAMA2 cut answers? Fix this
        return run_llm(prompt, stop=["</s>"]).strip()

def run_rag_chatgpt(prompt_input, search, client):
    with st.spinner("Retrieving results..."):

        db = get_faiss_vectordb(inference_api_key=os.getenv('INFERENCE_API_KEY'))

        # load the FAISS vector database from the generated index path
        # retrieve the top 5 most similar documents
        docs_and_scores = db.similarity_search_with_score(search)[:5]
        # print("\n\ndocs_and_scores: ", docs_and_scores, "\n\n")

        prompt = "Answer a question using References. Remember to mention the source(s) you use at the end of your answer. For example, if 'Source: 01_introduction.pdf, Page: 8', you can say 'slides 01_introduction.pdf, page 8'."
        user = f"Question: {prompt_input}\nReferences: "
        for doc, score in docs_and_scores:
            user += f"{doc} "

        user += "\nAnswer: "

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user}
        ]
        # print("\n\nmessages_rag: ", messages, "\n\n")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        ).choices[0].message.content

        return response

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    prompt = "Assist a student who is taking a university course. When you respond, you may respond on the knowledge you have or you may perform an Action, ONLY if necessary, i.e., the student asks for information about the course material. Action can be of one type only: (1) Search[content], which searches for similar content in the course material and returns the most relevant results if they exist. Put what you want to search for inside brackets after Search, like this: Search[What is the exam like?]. Be sure to put something that is inherent to the student's question."
    # prompt = "Assist a student who is taking a university course. If the student requests information about the course material that you're unsure of, you may perform a search action. Use the following format for a search action: Search[content], which searches for similar content in the course material. Replace content in brackets after Search with something inherent to the student's question. For example, Search[What is the exam like?] if the student asks about the exam. Only initiate a search when the student asks for information that is not within the scope of your knowledge."
    # examples = "\n<s>[INST] Hi, Im Bob! [/INST] Hi Bob, how can I help you today? [INST] I need information about the exam. Can you help me? [/INST] Sure, what do you want to know about the exam? [INST] How is the exam composed? [/INST] Search[How is the exam composed?]</s>"
    # prompt += examples

    string_dialogue = chat_completion_llama2(prompt)
    
    query = f"{string_dialogue}"
    output = run_llm(query=query, stop=["</s>"]).strip()

    # extract the Text if the Search[Text] action is performed
    if "Search[" in output:
        search = output.split("Search[")[1].split("]")[0]
        results = run_rag_llama2(prompt_input, search)
        output = "Search[" + search + "]\n\n" + results

    return output

def generate_chatgpt_response(prompt_input, client):
    # prompt = "Assist a student who is taking a university course. When you respond, you may respond on the knowledge you have or you may perform an Action, ONLY if necessary, i.e., the student asks for information about the course material. Action can be of one type only: (1) Search[content], which searches for similar content in the course material and returns the most relevant results if they exist. Put what you want to search for inside brackets after Search, like this: Search[What is the exam like?]. Be sure to put something that is inherent to the student's question."
    prompt = "You are a helpful assistant helping a student who is taking a university course. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Reply to student's requests information about the course with an Action. Action can be of one type:\n(1) Search[content], which searches for similar content in the course material.\nHere are some examples."
    examples = "\nUser: Hi, Im Bob!\nAssistant: Hi Bob, how can I help you today?\nUser: I need information about the exam. Can you help me?\nAssistant: Sure, what do you want to know about the exam?\nUser: How is the exam composed?\nAssistant: Search[exam format]"
    prompt += examples

    messages = chat_completion_chatgpt(prompt)
    messages[-1]["content"] += "\nAssistant: "
    # print("\n\nmessages: ", messages, "\n\n")
    output = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    ).choices[0].message.content

    # extract the Text if the Search[Text] action is performed
    if "Search[" in output:
        search = output.split("Search[")[1].split("]")[0]
        results = run_rag_chatgpt(prompt_input, search, client)
        output = "Search[" + search + "]\n\n" + results

    return output

# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if model_choice == "Llama":
                response = generate_llama2_response(prompt)
            else:
                client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
                response = generate_chatgpt_response(prompt, client)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    # print("\n\nsession_state.messages: ", st.session_state.messages, "\n\n")

# Refresh FAISS vectorstore every day at 03:00
def refresh_vectorstore():
    # delete the FAISS vector database
    get_faiss_vectordb(inference_api_key=os.getenv('INFERENCE_API_KEY'), refresh=True)

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Schedule fetch courses job to run every day at 03:00
scheduler = BackgroundScheduler(daemon=True)
trigger = CronTrigger(
    year="*", month="*", day="*", hour="3", minute="0", second="0"
)

scheduler.add_job(
    refresh_vectorstore,
    trigger=trigger,
    name="refresh_vectorstore",
)

scheduler.start()
