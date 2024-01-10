from io import StringIO

from pdfminer.layout import LTTextContainer

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from pdfminer.high_level import extract_pages

# from pdfminer.layout import LTTextContainer
# Create a file uploader

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title='LLM Summarization Docs')
st.title('🦜🔗 LLM Summarization')

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if "QA" not in st.session_state:
    st.session_state.QA = []

LLMDATA = {}


def create_documents(uploaded_file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000, length_function=len)
    text = []
    if ("txt" == uploaded_file.name.split(".")[-1]) or ("json" == uploaded_file.name.split(".")[-1]):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text.append(stringio.read())
    elif "pdf" == uploaded_file.name.split(".")[-1]:
        pdf_text=""
        for page_layout in extract_pages(uploaded_file):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    pdf_text+=element.get_text()
        text.append(pdf_text)
    else:
        raise Exception(f"File format {uploaded_file.name.split('.')[-1]} not supported")
    return text_splitter.create_documents(text)


def set_LLM(uploaded_file):
    global LLMDATA
    if uploaded_file is not None:
        filename = uploaded_file.name
        st.session_state["current_filename"] = filename
        if filename not in LLMDATA:
            print("uploaded_file(name)>>>>>>>>>", filename)
            # with open(filename) as f:
            #     state_of_the_union = f.read()
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000,length_function = len)
            documents = create_documents(uploaded_file)
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(documents, embeddings)
            LLMDATA[filename] = {
                "db": db
            }
            st.session_state['LLMDATA'] = LLMDATA
            print(len(LLMDATA))


def generate_response(query_text, filename):
    try:
        # Trying to access the 'LLMDATA' attribute in the 'session_state' object
        LLMDATA = st.session_state.LLMDATA
    except AttributeError:
        # Handling the AttributeError
        st.write("Please submit the uploaded file.")
        # You can choose to perform alternative actions here if needed
    except Exception as e:
        # Handling any other exceptions
        st.write(f"An unexpected error occurred: {e}")
        raise e
    if filename in LLMDATA:
        db = LLMDATA[filename]["db"]
        system_template = """
        You are an intelligent clinical trail researcher and excellent at finding answers from the documents.
        I will ask questions from the documents and you'll help me try finding the answers from it.
        Give the answer using best of your knowledge, say you dont know if not able to answer.
        ---------------
        {context}
        """
        qa_prompt = PromptTemplate.from_template(system_template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, model_name="gpt-4"), db.as_retriever(),
                                                   memory=memory, condense_question_prompt=qa_prompt)
        result = qa({"question": query_text})
        # return result["answer"]   
        dict = {"question": result["question"], "answer": result["answer"]}
        st.session_state.QA.append(dict)


def file_upload_form():
    with st.form('fileform'):
        supported_file_types = ["pdf", "txt", "json"]
        uploaded_file = st.file_uploader("Upload a file", type=(supported_file_types))
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.session_state.uploaded_file = uploaded_file
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_types:
                    set_LLM(uploaded_file)
                    st.session_state.current_filename = uploaded_file.name
                else:
                    st.write(f"Supported file types are {', '.join(supported_file_types)}")
            else:
                st.write("Please select a file to upload first!")


def query_form():
    with st.form('myform'):
        query_text = st.text_input('Enter your question:', placeholder='Enter your question here')
        submitted = st.form_submit_button('Submit', disabled=(query_text == ""))
        if submitted:
            filename = st.session_state.current_filename
            with st.spinner('Generating...'):
                generate_response(query_text, filename)
                for i in st.session_state.QA:
                    st.write("Question : " + i["question"])
                    st.write("Answer : " + i["answer"])


file_upload_form()
query_form()
