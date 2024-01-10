import streamlit as st
from streamlit_chat import message
import openai
import os

# get your API key from https://openai.com/ and set it as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="ChatGPT Business Writing Assistant",
    page_icon=":robot:"
)
st.markdown("<h1 style='text-align: center; color: red;'>ChatGPT Business Writing Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; width: auto'><a href='https://github.com'>Github Repo</a></div>",
            unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def get_completion(user_query):
    # system message is the first message in the conversation
    # that should give ChatGPT an idea of what's going on
    messages = [{"role": "system",
                 "content": "You are a helpful writing assistant. When I send you a message, improve it for a formal "
                            "business conversation."}]

    # requests can use up to 4097 tokens (~3000 words) shared between prompt and completion,
    # so we need to truncate the messages up to 2000 words to leave some room for the completion
    num_words = len(messages[0]["content"].split()) + len(user_query.split())
    for i in range(len(st.session_state['messages']) - 1, -1, -1):
        num_words += len(st.session_state['messages'][i]["content"].split())
        if num_words < 2000:
            messages.insert(1, st.session_state['messages'][i])
        else:
            break

    # append user query
    if user_query:
        messages.append({"role": "user", "content": user_query})

    # generate completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,  # 0-1, higher is more creative
    )

    return completion.choices[0].message.content.strip("\n")


# get user input query
# better add additional instructions so the model remembers the context
user_input = st.text_input("You: ", "Improve this message: ", key="input")
first_run = st.session_state.get("first_run", True)
if user_input and not first_run:
    output = get_completion(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": output})

# render messages
if st.session_state['messages']:
    for i in range(len(st.session_state['messages']) - 1, -1, -1):
        if st.session_state['messages'][i]['role'] == 'user':
            message(st.session_state['messages'][i]['content'], is_user=True, key=str(i) + '_user')
        else:
            message(st.session_state['messages'][i]['content'], key=str(i), seed=30)
else:
    # generate welcome message
    output = get_completion("")
    st.session_state["messages"].append({"role": "assistant", "content": output})
    message(output, key=str(0), seed=30)

st.session_state["first_run"] = False
