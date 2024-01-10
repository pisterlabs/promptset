import streamlit as st
from openai import OpenAI
import time

st.sidebar.title('🤖 Assistant')
avatar = {"assistant": "🤖", "user": "🐱"}
client = OpenAI(
    api_key=st.secrets['OPEN_AI_KEY'],
)
@st.cache_resource
def create_assistant():
    return client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )

assistant = create_assistant()

st.sidebar.write('## Assistant ID')
st.sidebar.write(assistant.id)

# Initialization
if 'thread' not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread = thread

thread = st.session_state.thread
st.sidebar.write('## Thread ID')
st.sidebar.write(thread.id)

# if st.sidebar.button('Delete Thread'):
#     client.beta.threads.delete(thread.id)

st.sidebar.write('## Prompt example')
st.sidebar.write("I need to solve the equation `3x + 11 = 14`. Can you help me?")

if prompt := st.chat_input():

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )

    # thread_messages = client.beta.threads.messages.list(thread_id=thread.id)
    # st.sidebar.write(thread_messages.data)

    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    time.sleep(5)

    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    # st.write(messages.data[::-1])
    for line in messages.data[::-1]:
        st.chat_message(line.role,avatar=avatar[line.role]).write(line.content[0].text.value)
