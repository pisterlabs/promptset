import streamlit as st
import hashlib
from dotenv import load_dotenv, find_dotenv
from langchain.llms import LlamaCpp

from document_embeddings import get_answer_with_documents
from models import init_messages, convert_langchain_schema_to_dict, create_llama2
from llama2_utils import llama2_prompt
from langchain.schema import (HumanMessage, AIMessage)
from session import Session

hide_menu_style = """
<style>
#MainMenu { visibility: hidden; }
footer  {
    visibility: hidden; 
}
</style>
"""
hide_menu_style2 = """
<style>
footer  {
    visibility: hidden; 
}
</style>
"""
st.markdown(hide_menu_style2, unsafe_allow_html=True)

session = Session()


class Scene:
    Login = 1
    Question = 2
    Feedback = 3
    Suggestion = 4


# Convert Pass into hash format
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# Check password matches during login
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False


def is_authentication():
    return session.contains("authentication_status")


def is_scene(scene: int):
    return session["authentication_status"] == scene


def set_scene(scene: Scene):
    session["authentication_status"] = scene


def set_authentication():
    session["authentication_status"] = Scene.Question


def login(username, password):
    hash = make_hashes("123")
    if check_hashes(password, hash):
        set_authentication()
        return True
    return False


def get_answer(llm, messages) -> str:
    # if isinstance(llm, ChatOpenAI):
    #     with get_openai_callback() as cb:
    #         answer = llm(messages)
    #     return answer.content, cb.total_cost
    if isinstance(llm, LlamaCpp):
        return llm(llama2_prompt(convert_langchain_schema_to_dict(messages)))


def show_login_form():
    if is_authentication():
        return
    login_form = st.empty()
    with login_form.form(key="login"):
        st.subheader('Log in to the App')
        username = st.text_input("User Name", placeholder='username')
        password = st.text_input("Password", type='password')
        submit_form = st.form_submit_button("Login")
        if submit_form:
            if login(username, password):
                login_form.empty()
            else:
                st.error("login fail")


def show_assistant_typing_answer(llm):
    with st.spinner("ChatGPT is typing ..."):
        answer = get_answer(llm, st.session_state.messages)
    show_assistant_message(answer)
    return answer


def show_assistant_typing_answer_stream(user_query):
    with st.spinner("ChatGPT is typing ..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            llm = create_llama2(message_placeholder)
            # full_response = get_answer(llm, st.session_state.messages)
            full_response = get_answer_with_documents(llm, user_query, st.session_state.messages)
        # full_response = ""
        # for response in llm_stream:
        #     # full_response += response.choices[0].delta.get("content", "")
        #     full_response += f"{response}"
        #     print(f"{full_response=}")
        #     message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)
    message_placeholder.markdown(full_response)
    return full_response


def show_chat_input():
    if not is_scene(Scene.Question):
        return
    if user_query := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_query))
        show_user_message(user_query)
        # answer = show_assistant_typing_answer(llm)
        answer = show_assistant_typing_answer_stream(user_query)
        st.session_state.messages.append(AIMessage(content=answer))
        return


def show_user_message(message: str):
    with st.chat_message("user"):
        st.markdown(message)


def show_assistant_message(message: str):
    with st.chat_message("assistant"):
        st.markdown(message)


def show_chat_message_history():
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            show_assistant_message(message.content)
        elif isinstance(message, HumanMessage):
            show_user_message(message.content)


def main():
    print("main")
    _ = load_dotenv(find_dotenv())
    init_messages()

    show_login_form()
    show_chat_message_history()
    show_chat_input()

    # if st.session_state["authentication_status"]:
    #     try:
    #         if auth.reset_password(st.session_state["username"], 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)


if __name__ == '__main__':
    main()
    print("=== END ===")
    