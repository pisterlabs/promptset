import os

import dotenv
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

from chatui.interactions.example_prompts import _run_example, generate_random_prompt
from chatui.interactions.salutations import get_time_based_greeting
from chatui.utils import get_project_root
from chatui.interactions.stream_handler import StreamHandler




def set_model() -> str:
    cola, colb, colc = st.columns(3)

    with colb:
        model_version = st.toggle(
            "Use GPT-4",
            value=False,
            key="model_version",
            help="Use GPT-4 instead of GPT-3",
        )

    if model_version:
        return "gpt-4"
    else:
        return "gpt-3.5-turbo"


def set_streamlit_config():
    st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
    st.title("📖 StreamlitChatMessageHistory")


def get_and_show_examples(_llm_chain):
    """show examples of what the model can do"""
    col1, col2, col3 = st.columns(3)

    for col in [col1, col2, col3]:
        examples = generate_random_prompt()

        with col:
            example = examples[-1]
            examples.pop()
            _run_example(_llm_chain, key=str(col), user_input=example)


def display_intro_message():
    """
    Display the introductory message and source code link.
    """
    st.markdown(
        """
    A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
    The messages are stored in Session State across re-runs automatically. View the
    [source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
    """
    )


def initialize_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message(f"{get_time_based_greeting()} How can I support you?")
    return msgs, memory


def fetch_openai_key():
    dotenv.load_dotenv(get_project_root() / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception("The OpenAI API-Keywas not found. Please check your .env-file.")
    return openai_api_key


def get_llm_chain(api_key, memory, model):
    template = """You are an polite, funny and helpful AI chatbot having a conversation with a human.

    {history}
    Human: {human_input}
    AI: """
    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )
    return LLMChain(
        llm=ChatOpenAI(
            openai_api_key=api_key, streaming=True, model=model
        ),
        prompt=prompt,
        memory=memory,
        verbose=True
    )


def render_chat_messages(msgs):
    for msg in msgs.messages:
        st.chat_message(msg.type).markdown(msg.content)


def render_chat(msgs):
    for msg in msgs.messages:
        if msg.type == "human":
            with st.chat_message("human", avatar="👤"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            with st.chat_message("ai", avatar="🤖"):
                st.markdown(msg.content)
        else:
            st.write(msg.type)
            st.markdown(msg.content)


def handle_user_input(llm_chain, msgs):
    user_input = st.chat_input()
    if user_input:
        # Prüfung auf Magic Commands

        if user_input.startswith("/qa"):
            msgs.add_ai_message("`/qa:` command detected")

        elif user_input.startswith("/img"):
            msgs.add_ai_message("`/img:` command detected")

        elif user_input.startswith("/"):
            msgs.add_system_message(
                """ __List of commands:__ \
                - `/img:` command detected \
                - `/qa:` command detected"""
            )

        else:
            llm_chain.run(user_input)


def display_memory_contents():
    with st.expander("View the message contents in session state"):
        st.markdown(
            """
        Memory initialized with:
        ```python
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        memory = ConversationBufferMemory(chat_memory=msgs)
        ```

        Contents of `st.session_state.langchain_messages`:
        """
        )
        st.json(st.session_state.langchain_messages)
        st.write(st.session_state.langchain_messages[0])
        st.write(type(st.session_state.langchain_messages[0]))


def main():
    set_streamlit_config()
    model = set_model()
    msgs, memory = initialize_memory()
    api_key = fetch_openai_key()
    llm_chain = get_llm_chain(api_key, memory, model)
    get_and_show_examples(llm_chain)
    handle_user_input(llm_chain, msgs)
    display_memory_contents()
    # render_chat_messages(msgs)
    render_chat(msgs)


if __name__ == "__main__":
    main()
