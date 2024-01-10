import streamlit as st 
import os
import time
from utils import getPages, splitPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

# Initialize session state for conversation history and assistant ID
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'assistant_id' not in st.session_state:
    st.session_state.assistant_id = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None

def main():
    st.title('Teaching-Learning System with OpenAI Assistant API')

    # User inputs their own OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    api_key_entered = api_key != ""

    # Initialize client only if API key is entered
    if api_key_entered:
        client = OpenAI(api_key=api_key)

    # The other input fields are always visible but disabled if no API key is entered
    subject = st.text_input("Subject of the Document:", disabled=not api_key_entered)
    document_file = st.file_uploader("Upload your educational material (PDF):", type=['pdf'], disabled=not api_key_entered)
    question = st.text_input("Your Initial Question:", disabled=not api_key_entered)

    if api_key_entered and st.button("Send question"):
        if subject and document_file and question:
            with st.spinner("Processing..."):
                process_document(subject, document_file, question, client)
        else:
            st.warning("Please provide the subject, upload a document, and enter a question.")
    
    # Display conversation
    for idx, message in enumerate(st.session_state.conversation):
        st.markdown(f"**Message {idx + 1}:**\n{message}")

    # Show the chat input only after the first response has been received
    if api_key_entered and len(st.session_state.conversation) > 0:
        user_message = st.text_input("Continue the Conversation:", key="user_input")
        send_button = st.button("Send Message", on_click=handle_send_message, args=(user_message,client))


def handle_send_message(message):
    if message:
        handle_user_message(message)
        st.session_state.user_input = ''  # Reset the input box


def process_document(subject, document_file, question, client):
    if document_file is not None:
        with open("temp_document.pdf", "wb") as f:
            f.write(document_file.getbuffer())
        document_path = "temp_document.pdf"

        # Document processing logic
        pages = getPages(document_path)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = [page.page_content for page in pages]
        text_ids = [str(page.metadata['page']) for page in pages]
        retriever = Chroma.from_texts(texts, embedding=embeddings, ids=text_ids).as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        text_dict = dict(zip(texts, text_ids))
        relevant_page = int(text_dict[docs[0].page_content])
        assistant_document = splitPDF(document_path, relevant_page)

        # Start the conversation with the initial question
        thread_id, assistant_id = create_assistant_and_thread(subject, document_path, client)
        st.session_state.assistant_id = assistant_id
        st.session_state.thread_id = thread_id
        handle_user_message(question, client)

def handle_user_message(message, client):
    st.session_state.conversation.append(f"You: {message}")
    assistant_response = get_assistant_response(message, client)
    st.session_state.conversation.append(f"Assistant: {assistant_response}")

def get_assistant_response(user_message, client):
    thread_id = st.session_state.thread_id
    assistant_id = st.session_state.assistant_id

    message = client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_message)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    run = run_and_wait(run, thread_id, client)
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value

def run_and_wait(run, thread_id, client):
    start_time = time.time()
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        time.sleep(1/4)
    return run

def create_assistant_and_thread(subject, document_path, client):
    file = client.files.create(file=open(document_path, "rb"), purpose='assistants')
    assistant = client.beta.assistants.create(
        name="Teacher Assistant",
        instructions=f'''You are a teacher in the {subject}. Given the context information and your prior knowledge, generate an appropriate guidance based on the question and the information from the source material.
        
        Your guidance should be step by step, such that you wait for a student to help answer its own question. In a way it should be socratic dialogue based. 

        Your answers should be guiding, trying to help the student learn. Don't give the answers away, guide the student iteratively. Simplify the procedure step by step and wait for the student's responses.''',
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )
    thread = client.beta.threads.create()
    return thread.id, assistant.id

if __name__ == "__main__":
    main()