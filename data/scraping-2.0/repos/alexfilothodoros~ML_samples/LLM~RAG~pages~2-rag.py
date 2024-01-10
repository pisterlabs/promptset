import os
import PyPDF2
import streamlit as st
from typing import Any, Dict, List
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from dotenv import load_dotenv
from glob import glob

st.set_page_config(page_title="EPLAN", page_icon="📚", layout="wide")
load_dotenv(dotenv_path="../.env")

open_api_key = os.getenv("OPENAI_API_KEY")


@st.cache_data
def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        else:
            st.warning("Please provide a pdf file.", icon="⚠️")

    return all_text


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever


@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("No splits created. Please try again.")
        st.stop()

    return splits


def main():
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
           </style>
        """,
        unsafe_allow_html=True,
    )

    retriever_type = "SIMILARITY SEARCH"
    splitter_type = "RecursiveCharacterTextSplitter"

    if "openai_api_key" not in st.session_state:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    uploaded_files = st.file_uploader(
        "Upload a PDF Document", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.success("Files have been uploaded")
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if (
            "last_uploaded_files" not in st.session_state
            or st.session_state.last_uploaded_files != uploaded_files
        ):
            st.session_state.last_uploaded_files = uploaded_files
            if "eval_set" in st.session_state:
                del st.session_state["eval_set"]
        loaded_text = load_docs(uploaded_files)
        splits = split_texts(
            loaded_text, chunk_size=1000, overlap=100, split_method=splitter_type
        )
        embeddings = OpenAIEmbeddings()
        retriever = create_retriever(embeddings, splits, retriever_type)
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        chat_openai = ChatOpenAI(
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
            temperature=0,
        )
        qa = RetrievalQA.from_chain_type(
            llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True
        )
        example_switch = st.sidebar.toggle("Use example questions")
        if not example_switch:
            user_question = st.text_input("Enter your question:")
        elif example_switch:
            example_question = st.sidebar.selectbox(
                "example questions",
                [
                    "Does the license allow commercial use?",
                    "Which license are we reading?",
                    "What is a license?",
                    "What is a license?",
                    "What is a leagal entity?",
                    "How can I apply this license to my work?",
                ],
            )
            user_question = example_question

        if user_question:
            answer = qa.run(user_question)
            st.write(f"{answer}")


if __name__ == "__main__":
    main()
