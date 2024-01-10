import os
import base64
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory


def show_pdf(file_buffer):
    base64_pdf = base64.b64encode(file_buffer.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def show_md(file_buffer):
    content = file_buffer.read().decode()
    st.markdown(content)


def extract_text(uploaded_file, file_extension):
    text = ""
    if file_extension == '.pdf':
        show_pdf(uploaded_file)
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == '.md':
        show_md(uploaded_file)
        text = uploaded_file.read().decode()

    return text


def process_text(text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def answer_question(docs, user_question, useAzure = False):
    # add Azure gpt-4 api support in the future
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    chat = ChatOpenAI(temperature=0.1)

    messages = [SystemMessage(
        content="You are a helpful bot. If needed, you can use the following document to help you answer the question.")]

    # Add the relevant documents to the chat context
    for i, doc in enumerate(docs):
        messages.append(SystemMessage(content=f"Document {i+1}: {doc}"))

    messages.append(HumanMessage(content=user_question))

    response = chat(messages).content
    messages.append(AIMessage(content=response))

    st.session_state.memory.save_context(
        {"input": user_question},
        {"ouput": response}
    )

    return response

def display_chat_history(chat_history_container):
    with chat_history_container:
        chat_history_html = "<div style='height: 500px; overflow-y: scroll;'>"
        st.header("Chat History")
        for msg in st.session_state.chat_history:
            chat_history_html += (
                f"<div style='text-align: right; color: blue;'>You: {msg['user']}</div>"
            )
            chat_history_html += (
                f"<div style='text-align: left; color: green;'>GPT-3.5: {msg['response']}</div>"
            )
        
        chat_history_html += "</div>"
        
        st.write(chat_history_html, unsafe_allow_html=True)


def display_selected_documents(col, docs):
    if not docs:
        return

    for idx, doc in enumerate(docs):
        title = f"Document {idx+1}"
        with col.expander(title):
            st.write(doc.page_content)



def main():
    st.set_page_config(page_title="Ask your PDF", layout="wide")

    load_dotenv()
    st.header("Ask your PDF 💬")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    col1, col2 = st.columns([0.6, 0.4])  # Adjust column width

    # Initialize docs and selected_documents
    docs = []

    with col1:  # Move the file_uploader into the col1 context
        uploaded_file = st.file_uploader("Choose a PDF or Markdown file", type=['pdf', 'md'], key="file_uploader")

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            # Display the uploaded file on the left side
            with col1.expander("Uploaded File"):
                text = extract_text(uploaded_file, file_extension)

            knowledge_base = process_text(text)

        # Initialize chat history container
        chat_history_container = col2.empty()

        # Store chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show user input
        user_question = col2.text_input("Ask a question about your PDF:")

        # Send button in column 2
        if col2.button("Send"):
            docs = knowledge_base.similarity_search(user_question)
            response = answer_question(docs, user_question, useAzure=False)
            st.session_state.chat_history.append({"user": user_question, "response": response})

            # Clear user input
            user_question = ""


        # Display document 
        display_selected_documents(col2, docs)

        display_chat_history(chat_history_container)

if __name__ == "__main__":
    main()

