import time
import streamlit as st
from docai import (
    get_vectorstore,
    jaccard_similarity_list,
    read_pdf_with_pypdf2,
    get_documents,
    generate_pdf,
    generate_diff_pdf
)
from loguru import logger
from openai_helper import converse, populate_chat_history, extract_questions
import hashlib
import base64


class ChatHistory:

    def __init__(self):
        self.message_list = []
        self.chat_history = {}

    def append_message(self, new_chat):
        self.message_list.append(new_chat)
        return self.message_list

    def format_message(self, question, answer):
        return (question, answer, [])

    def convert_msg_to_chat_history(self):
        self.chat_history = populate_chat_history(
            message_list=self.message_list)


class ChatManager:
    def generate_question(self, question, chat_history={}):
        if chat_history is {}:
            return question
        return extract_questions(text=question, model="gpt-3.5-turbo", conversation_log=chat_history)

    def chat(self, references, query, model="gpt-3.5-turbo", conversational_log={}, temperature=0.01):
        response_gen = converse(references=references,
                                user_query=query,
                                model=model,
                                conversation_log=conversational_log,
                                temperature=temperature)
        chat_response = response_gen.content
        return chat_response


class DocAI:
    def __init__(self, doc_id, documents):
        self.doc_id = doc_id
        self.vectorstore = get_vectorstore(documents)

    def retrieve_top_k_documents(self, question, top_k):
        # self.top_docs = self.vectorstore.similarity_search(query=question, k=top_k)
        doc_val = {score: doc for doc, score in self.vectorstore.similarity_search_with_score(
            query=question, k=top_k)}
        sorted_doc_values = sorted(list(doc_val.keys()), reverse=False)
        logger.debug(f"docs score : {sorted_doc_values}")
        threshold = 0.02
        top_score = sorted_doc_values[0]
        top_doc = []
        for score in sorted_doc_values:
            logger.debug(f"score -> {score}")
            if score <= top_score+threshold:
                top_doc.append(doc_val[score])
            else:
                break
        self.top_docs = top_doc
        logger.debug(f"docs store : {self.top_docs}")
        return self.top_docs

def main(documents):
    docai = DocAI(doc_id="123", documents=documents)
    chat_manager = ChatManager()
    chat_history = ChatHistory()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.memory = None

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # if st.session_state.memory:
    #     docai.set_memory(st.session_state.memory)

    # Altering the memory state of Human message with generated question
    if st.session_state.memory:
        chat_msg = st.session_state.memory
        msg_list = chat_history.append_message(chat_msg)
        chat_history.convert_msg_to_chat_history()

    # Reclaim the memory

    # React to user input
    if prompt := st.chat_input("Ask me anything"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # if len(prompt.split(" ")) < 3:
        #     prompt = "What is meant by " + prompt

        question = prompt.strip()
        content = docai.retrieve_top_k_documents(question=question, top_k=3)
        logger.debug(f"Retrieve Content -> {content}")
        extracted_chunks_text = [
            extracted_chunk.page_content for extracted_chunk in content]
        generated_question = chat_manager.generate_question(
            question=question, chat_history=chat_history.chat_history)
        logger.debug(f"Generated question -> {generated_question}")
        logger.debug(f"Chat History -> {chat_history.chat_history}")
        answer = chat_manager.chat(
            references=extracted_chunks_text, query=generated_question)
        # answer = answer_resp.split("### compiled references:")[0]
        logger.debug(f"Answer_resp: {answer}")

        # Save chat history
        new_chat = chat_history.format_message(question, answer)
        st.session_state.memory = new_chat

        # Calculating Jaccard Similarity score of response with extracted chunk
        scores = jaccard_similarity_list(
            resp=answer, extracted_chunk=extracted_chunks_text)
        logger.debug(f"Scores -> {scores}")
        logger.debug(f"Answer: {answer}")
        response = f"{answer}"
        # logger.debug(f"response: {response}")
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


def calculate_hash(file_content):
    sha1 = hashlib.sha1()
    sha1.update(file_content.encode("utf-8"))
    return sha1.hexdigest()


def download_link(pdf_bytes):
    # Display a download link
    st.markdown(
        f'<a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" download="downloaded_pdf.pdf">Click here to download the PDF</a>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Generate a unique key based on time
    # Add custom CSS styles
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            color: #000000;
            margin-bottom: 2px;
        }
        .info {
            font-size: 18px;
            color: #fffff;
            margin-bottom: 30px;
        }
        .custom-button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # st.title("ChatPDF")
    st.markdown('<p class="title"><b>ChatPDF</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="info">Upload PDF to get started with ChatPDF</p>',
                unsafe_allow_html=True)
    unique_key = f"file_uploader_{time.time()}"
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file here...", type=['pdf'])
    if uploaded_file is not None:

        # links = uploaded_file.read().decode("utf-8")
        pdf_reader_text = read_pdf_with_pypdf2(uploaded_file)
        previous_hash = st.session_state.get("file_hash", None)
        current_hash = calculate_hash(pdf_reader_text)
        if previous_hash is None or current_hash != previous_hash:
            st.session_state.file_hash = current_hash
            st.session_state.messages = []
            st.session_state.memory = None
        # logger.debug(f"uploaded files: {pdf_reader_text}")
        documents = get_documents(pdf_reader_text)
        main(documents)
        # if st.sidebar.button("Download PDF"):
        # Create a PDF and offer it for download
        # pdf_bytes = generate_pdf(st.session_state.messages)
        # download_link(pdf_bytes)
        if st.session_state.messages and \
            st.sidebar.download_button("Download PDF",
                                       generate_pdf(st.session_state.messages),
                                       file_name='downloaded_pdf.pdf', key='download_button'):
            pass
        # if st.session_state.messages and \
        #     st.sidebar.download_button("Download Diff PDF",
        #                                generate_diff_pdf(
        #                                    st.session_state.messages, pdf_reader_text),
        #                                file_name='downloaded_diff_pdf.pdf', key='download_diff_button'):
        #     pass
