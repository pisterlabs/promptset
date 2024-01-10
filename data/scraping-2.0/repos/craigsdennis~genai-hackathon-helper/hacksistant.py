# Graciously lifted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

st.title("Hacksistant")

""""
This is a work in progress that will attempt to use the beta Assistant API
Inside Streamlit.
TODO:

- [x] Choose Assistant
- [x] Limit list of available assistants using Metadata
- [ ] Create a new Thread
- [ ] Use a Run
- [ ] Display old messages in Thread
- [ ] Add File uploading for entire Assistant
- [ ] Add File uploading for Thread

"""


@st.cache_resource
def get_client():
    return OpenAI()


client = get_client()


@st.cache_data()
def get_assistants():
    all = client.beta.assistants.list(order="desc")
    return list(filter(lambda a: a.metadata.get("for", None) == "hacksistant", all))

def choose_assistant():
    st.session_state.assistant = st.session_state["assistant_choice"]

# If one isn't chosen in state
if "assistant" not in st.session_state:
    assistants = get_assistants()
    print(assistants)
    if assistants:
        # Show list of assistants, key is the name
        assistant_choice = st.selectbox(
            "Choose your assistant",
            key="assistant_choice", 
            options=assistants, 
            format_func=lambda a: a.name,
            on_change=choose_assistant
        )
    st.write("OR Create a new one")
    with st.form("create-assistant"):
        name = st.text_input("Name")
        description = st.text_input("Description")
        instructions = st.text_area("Instructions")
        model = st.selectbox("Model", options=["gpt-3.5-turbo", "gpt-4"])
        submitted = st.form_submit_button("Add Assistant")
        if submitted:
            response = client.beta.assistants.create(
                name=name,
                description=description,
                instructions=instructions,
                model=model,
                metadata={"for": "hacksistant"},
            )
            st.cache_data.clear()
            st.session_state["assistant"] = response
    st.stop()

# Now that the assistant is chosen
assistant = st.session_state.assistant
with st.sidebar:
    # TODO: add editor
    with st.form("update-assistant"):
        name = st.text_input("Name", value=assistant.name)
        description = st.text_input("Description", value=assistant.description)
        instructions = st.text_area("Instructions", value=assistant.instructions)
        model = st.selectbox(
            "Model", options=["gpt-3.5-turbo", "gpt-4"], value=assistant.model
        )
        submitted = st.form_submit_button("Update assistant")
        if submitted:
            response = client.beta.assistants.update(
                assistant_id=assistant.id,
                description=description,
                instructions=instructions,
                model=model,
            )
            st.session_state.assistant = response

