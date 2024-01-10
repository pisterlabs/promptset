from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
import streamlit as st

INITIAL_MESSAGE = "You are a helpful assistant."
PAGE_TITLE = "Chat with GPT-3"
PAGE_ICON = "🤖"
SIDEBAR_TITLE = "Options"
HEADER_TITLE = "Chat with ChatGPT 😘"
CHAT_PROMPT = "聞きたいことを入力してね！"
LOADING_MESSAGE = "ChatGPT is typing..."
MODELS = ("gpt-3.5-turbo", "gpt-4")

def init_chat() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.header(HEADER_TITLE)
    st.sidebar.title(SIDEBAR_TITLE)

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear chat history", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=INITIAL_MESSAGE)]
        st.session_state.costs = []

def select_model() -> tuple[str, float]:
    model = st.sidebar.selectbox("Select model", MODELS)
    model = "gpt-4" if model == "gpt-3.5-turbo" else model
    min = 0.0
    max = 2.0
    value = 0.9
    step = 0.01
    temperature = st.sidebar.slider("temperature:", min, max, value, step)
    return model, temperature

def handle_user_input(llm) -> None:
    if user_input := st.chat_input(CHAT_PROMPT):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner(LOADING_MESSAGE):
            response_content, total_cost = get_answer(llm, st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response_content))
            st.session_state.costs.append(total_cost)

def display_messages() -> None:
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

def get_answer(llm: str, messages: list) -> tuple[str, float]:
    with get_openai_callback() as callback:
        answer = llm(messages)
    return answer.content, callback.total_cost

def display_cost() -> None:
    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
