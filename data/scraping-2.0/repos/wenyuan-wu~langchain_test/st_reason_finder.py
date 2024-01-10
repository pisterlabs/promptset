import logging

import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from db_test import create_db

from reason_finder import get_full_info, create_context_template, reason_extractor

load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Prototype Zero of Agent",
    page_icon="🤖"
)

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "reason" not in st.session_state:
    st.session_state["reason"] = ""
if "prompt_context" not in st.session_state:
    st.session_state["prompt_context"] = ""
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
if "history" not in st.session_state:
    st.session_state["history"] = []


# Define function to get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        # what is this?
        st.session_state["input"],
        key="input",
        placeholder="Type 'Hi' to get started.",
        label_visibility="hidden",
    )
    return input_text


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    # range -1?
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.memory.buffer.clear()
    st.session_state.prompt_context = ""
    st.session_state.history = []
    st.session_state.reason = ""


# Set up the Streamlit app layout
st.write("# Prototype Zero of Agent 🤖")
st.markdown("Prototype of a conversational agent, which is triggered by a event and "
            "conduct conversation with patient to extract the reason of the event")

patient_db, plan_db, measure_db = create_db()
patient_df = pd.DataFrame.from_dict(patient_db, orient='index', columns=["name", "age", "BMI", "plan"])
st.markdown("## Patient Database")
st.write(patient_df)
plan_df = pd.DataFrame.from_dict(plan_db, orient='index', columns=["plan"])
st.markdown("## Plan Database")
st.write(plan_df)
measure_df = pd.DataFrame.from_dict(measure_db, orient='index', columns=["measure"])
st.markdown("## Measure Database")
st.write(measure_df)
event_json = {
    "patient": 12,
    "plan": 27,
    "measure": 18,
    "completed": False
}
st.markdown("## Event JSON")
st.write(event_json)
st.markdown("## Conversational Agent")

event = get_full_info(event_json)
st.session_state.prompt_context = create_context_template(event)

sys_template = "The following is a friendly conversation between a physician and a patient. " \
               "The physician is talkative and provides lots of specific details from its context." \
               "The physician is trying to find out why the patient did not complete the planned measure. " \
               "The first message from user input will be 'Hi', " \
               "ignore this message and create a short greeting to get the conversation started. " \
               "Here is the context information: "
sys_template = sys_template + st.session_state.prompt_context
# Create an OpenAI instance
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sys_template, validate_template=False),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Create the ConversationChain object with the specified configuration
Conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory,
)


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input and st.button("Submit", type="primary"):
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.history.append(f"patient: {user_input}")
    st.session_state.history.append(f"physician: {output}")
    logging.info(f"{st.session_state.history}")
    st.session_state.reason = reason_extractor(st.session_state.prompt_context, st.session_state.history)

# Set up sidebar with various options
with st.sidebar.expander("Status", expanded=True):
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.memory.buffer)
    st.markdown("## Preview prompt context")
    st.write(st.session_state.prompt_context)
    st.markdown("## Preview reason extractor")
    st.write(st.session_state.reason)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.success(st.session_state["generated"][i], icon="👨‍⚕️")
        st.info(st.session_state["past"][i], icon="🧐")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.button("Clear all", type="primary"):
        st.session_state.stored_session = []

st.markdown("---")

