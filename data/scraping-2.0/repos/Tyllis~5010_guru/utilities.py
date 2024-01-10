import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


def check_password(state, secret):
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if state["password"] in secret["password"]:
            state["password_correct"] = True
            del state["password"]  # don't store password
        else:
            state["password_correct"] = False

    if "password_correct" not in state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password",
            placeholder='***********',
            help="""
            Right key, you do not have. Master Jun, you must seek.
            """
        )
        return False
    elif not state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def get_database(state, secret):
    state.embeddings = OpenAIEmbeddings(openai_api_key=secret["openai_api"])
    state.database = FAISS.load_local("data", state.embeddings)


def get_llm(state, secret):
    state.llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=secret['openai_api']
    )


def rag_response(state, query):
    # Retrieval
    docs = state.database.similarity_search(query, k=5)
    context = '/n'.join([doc.page_content for doc in docs])
    source = ""
    for idx, doc in enumerate(docs):
        file_name = doc.metadata['file_name']
        file_link = doc.metadata['file_link']
        circ_name = doc.metadata['circular_name']
        start_page = doc.metadata['start_page']
        end_page = doc.metadata['end_page']
        source += f"Source #{idx + 1}\nDocument: {circ_name}\nPage Start: {start_page}\nPage End: {end_page}\n"
        source += f"File Name: {file_name}\nFile Link: {file_link}\n------------------------\n"

    # Generation
    template = f"Answer the question only based on the following context inside the triple hashtag."
    template += "If you cannot find the answer in the context just say you cannot "
    template += "find the answer in the available circulars. Do not mention the context."
    template += f"\n###\nContext:\n{context}\n###\nQuestion: {query}"
    prompt = [
        SystemMessage(
            content="You are a helpful assistant that is expert in FTA's circulars."
        ),
        HumanMessage(
            content=template
        ),
    ]
    response = state.llm(prompt).content.strip()

    # Create output source content
    content = ""
    for doc in docs:
        content += f"SOURCE: {doc.metadata['circular_name']}\nPAGE START: {doc.metadata['start_page']}\n"
        content += f"Page END: {doc.metadata['end_page']}\nCONTENT: \n{doc.page_content}\n"
        content += "------------------------\n"
    return response, source, content


def get_answer(state, query):
    state.text_error = ''
    if not query:
        state.text_error = "Please ask a question."
        return
    state.answer, state.source, state.content = rag_response(state, query)
    state.query = query
