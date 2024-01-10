import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from utils.utils import (
    extract_text_from_PDF,
    split_content_into_chunks,
    translate_into_chinese,
    summary_from_texts,
    save_chunks_into_vectorstore, search_from_content)


# Initialize llm, embedding model and vectorstore Object
llm = None
embedding_model = None
qa_chain = None

# Setting ChatPDF page
st.set_page_config(page_title="ChatPDF", layout="wide")
st.title("ChatPDF 📃")

if "LANGUAGE" not in st.session_state:
    st.session_state["LANGUAGE"] = "简体中文"

# set button string
if st.session_state["LANGUAGE"] == "简体中文":
    st.markdown(
        """
        #### 注意：
        1. 至少要在Setting页面设置OpenAI API Key，才能使用PDF翻译和内容摘要的功能。
        2. 此外，您还需要设置 Pinecone API Key 才能够使用提问PDF内容。
        3. 在使用ChatPDF功能时，请勿随意切换至其他页面（例如Chat，Setting等），ChatPDF不会缓存消息。
        4. 当前版本仅支持每次上传一个文件。
        """
    )
    file_uploader_string = "上传PDF文件，点击‘提交并处理’"
    submit_button_string = "提交并处理"
    submit_button_spinner_string = "稍等..."
    tab_translation_string = "翻译"
    tab_summarization_string = "总结"
    translate_button_string = "翻译！"
    translate_button_spinner_string = "翻译中..."
    summary_button_string = "总结！"
    summary_button_spinner_string = "总结中..."
    chat_input_string = "问点儿什么吧..."
    qa_warning_string = "哎呀！ 如果你想测试 Q&A 功能，请先设置Pinecone API Key 和提交PDF文件"
    openai_error_string = "哎呀！你好像没设置OpenAI API Key。"
elif st.session_state["LANGUAGE"] == "English":
    st.markdown(
        """
        #### Note:
        1. To set OpenAI API Key in setting page at least, you can enjoy the PDF translation and content summarization features
        2. In addition, you need to set Pinecone API Key to ask question about the content of the PDF document.
        3. When using the ChatPDF function, please do not switch to other pages (e.g. Chat, Setting, etc.), ChatPDF does not cache messages.
        4. The current version only supports uploading one file at a time. 
        """
    )
    file_uploader_string = "Upload PDF file, click 'Submit and Process'"
    submit_button_string = "Submit and Process"
    submit_button_spinner_string = "Waiting..."
    tab_translation_string = "Translation"
    tab_summarization_string = "Summarization"
    translate_button_string = "Translate"
    translate_button_spinner_string = "Translating..."
    summary_button_string = "Summary"
    summary_button_spinner_string = "Summarizing..."
    chat_input_string = "Ask something..."
    qa_warning_string = "Oops! If you want to test Q&A, set valid Pinecone API Key and submit PDF file first."
    openai_error_string = "Oops! It seems the OpenAI API Key is invalid."

# switch page
if "switch_page" not in st.session_state:
    st.session_state["switch_page"] = "ChatPDF"

# Setting initialization
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
elif st.session_state["OPENAI_API_KEY"] != "":
    llm = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])
    embedding_model = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API_KEY"])

if "PINECONE_API_KEY" not in st.session_state:
    st.session_state["PINECONE_API_KEY"] = ""
if "PINECONE_ENVIRONMENT" not in st.session_state:
    st.session_state["PINECONE_ENVIRONMENT"] = ""

if "pdf_files" not in st.session_state:
    st.session_state["pdf_files"] = []
if "texts" not in st.session_state:
    st.session_state["texts"] = []
if "content_chunks" not in st.session_state:
    st.session_state["content_chunks"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = []

# pdf chat message
if "pdf_messages" not in st.session_state:
    st.session_state["pdf_messages"] = []

# if page switch, reset all pdf channel
if st.session_state["switch_page"] != "ChatPDF":
    st.session_state["pdf_files"] = []
    st.session_state["texts"] = []
    st.session_state["content_chunks"] = []
    st.session_state["vector_store"] = []
    st.session_state["pdf_messages"] = []

# Initialize texts and content_chunks Object
texts = st.session_state["texts"]
content_chunks = st.session_state["content_chunks"]

# TODO: Main
if llm:
    # upload the pdf document
    files = st.file_uploader(file_uploader_string,
                             accept_multiple_files=False)
    files_type = "PDF"
    # click the 'Summit and Process' button to finish document pretreatment
    submit_button = st.button(submit_button_string)
    if submit_button:
        with st.spinner(submit_button_spinner_string):
            st.session_state["pdf_files"] = files

            # 1. get PDF files
            texts = extract_text_from_PDF(files)
            st.session_state["texts"] = texts

            # 2. split pdf into chunk

            content_chunks = split_content_into_chunks(texts, files_type)
            st.session_state["content_chunks"] = content_chunks

    tab1, tab2 = st.tabs([tab_translation_string, tab_summarization_string])
    # Option 1: translation
    with tab1:
        if st.button(translate_button_string):
            with st.spinner(translate_button_spinner_string):
                translate_into_chinese(content_chunks, llm, files_type)
    # Option 2: summarization
    with tab2:
        if st.button(summary_button_string):
            with st.spinner(summary_button_spinner_string):
                summary_from_texts(content_chunks, llm, files_type)

    # Option 3: Q&A, ensure submit 1 doc 1 time
    if submit_button and st.session_state["PINECONE_API_KEY"] != "" and st.session_state["PINECONE_ENVIRONMENT"] != "":
        # Initialize and save chunks into vector store.
        namespace = "pdf"
        st.session_state["vector_store"] = save_chunks_into_vectorstore(content_chunks, embedding_model, namespace, files_type)

    if st.session_state["vector_store"]:

        for message in st.session_state["pdf_messages"]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        prompt = st.chat_input(placeholder=chat_input_string)
        search_from_content(llm, st.session_state["vector_store"], prompt, files_type)

    else:
        st.warning(qa_warning_string)

else:
    with st.container():
        st.error(openai_error_string)

st.session_state["switch_page"] = "ChatPDF"
