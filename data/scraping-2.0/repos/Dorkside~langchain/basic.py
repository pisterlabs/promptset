from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts import PromptTemplate

from config import set_env_vars

from models import StreamHandler, StreamlitChatMessageHistoryDB

import streamlit as st

from choices import CHOICES

set_env_vars()

st.set_page_config(page_title="Daz", page_icon="📖")
st.title("📖 Daz")

if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7

query_params = st.experimental_get_query_params()

if "conversation" in query_params:
    conversation_id = int(query_params["conversation"][0])
    msgs = StreamlitChatMessageHistoryDB(
        key="langchain_messages", conversation_id=conversation_id
    )
    # Load previous conversations from the database
    msgs.load_history()
else:
    msgs = StreamlitChatMessageHistoryDB(key="langchain_messages")


from sidebar import Sidebar

sb = Sidebar()

memory = ConversationBufferMemory(chat_memory=msgs)

view_messages = st.expander("View the message contents in session state")

if st.session_state.template:
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=CHOICES[st.session_state.template],
    )
chat = ChatOpenAI(
    streaming=True,
    callbacks=[],
    temperature=st.session_state.temperature,
    model="gpt-4-1106-preview",
)
if prompt is not None:
    llm_chain = LLMChain(llm=chat, prompt=prompt, memory=memory)
else:
    llm_chain = LLMChain(llm=chat, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


# If user inputs a new prompt, generate and draw a new response
if user_input := st.chat_input():
    st.chat_message("human").write(user_input)
    chat_box = st.chat_message("ai").empty()
    stream_handler = StreamHandler(chat_box, display_method="write")

    chat.callbacks = [stream_handler]

    # Note: new messages are saved to history automatically by Langchain during run
    try:
        llm_chain.run(user_input)
    except Exception as e:
        st.error(e)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
