import requests
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Load the .env file located in the project directory
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASANA_API_KEY = os.getenv('ASANA_API_KEY')
ASANA_PROJECT_GID = '1204150246906885'  # Trouble-shooting gid

def retrieve_tickets_from_asana():
    try:
        headers = {
            "Authorization": f"Bearer {ASANA_API_KEY}"
        }
        response = requests.get(f"https://app.asana.com/api/1.0/projects/{ASANA_PROJECT_GID}/tasks?opt_fields=name,notes,permalink_url", headers=headers)
        response.raise_for_status()
        tasks = response.json()["data"]

        tickets = []
        for task in tasks:
            ticket_name = task["name"]
            ticket_description = task["notes"]
            ticket_link = task["permalink_url"]
            tickets.append(f"Question: {ticket_name}\n\n Answer: {ticket_description}\n\n Refernce: {ticket_link}")
        return tickets
    except Exception as e:
        st.error(f"Error retrieving tickets from Asana: {e}")
        return []

def process_tickets(tickets):
    try:
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(tickets, embeddings)

        return knowledge_base
    
    except Exception as e:
        st.error(f"Error processing tickets: {e}")
        return None

def answer_question(docs, user_question):
    try:
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)

        chat = ChatOpenAI(temperature=0.1)

        messages = [SystemMessage(
            content='''
            You are a helpful bot. You need to teach customer how to solve the problem 
            according to the answer part of Document and provide the reference link at the end.
            You should answer the question in Traditional Chinese. The customer is from IT team.
            '''
            )]

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
    
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return "I'm sorry, I couldn't process your question. Please try again."


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
    st.set_page_config(page_title="Ask your FAQ", layout="wide")

    load_dotenv()
    st.header("Ask your FAQ 💬")

    # Initialize docs and selected_documents
    docs = []
    
    tickets = retrieve_tickets_from_asana()
    
    knowledge_base = process_tickets(tickets)

    col1, col2 = st.columns([0.4, 0.6])  # Adjust column width
    
    # Store chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with col1:
        # Show user input
        user_question = st.text_input("Ask a question about your FAQ:")

        # Send button
        if st.button("Send"):
            docs = knowledge_base.similarity_search(user_question)
            response = answer_question(docs, user_question)
            st.session_state.chat_history.append({"user": user_question, "response": response})

            # Clear user input
            user_question = ""

        # Display document 
        display_selected_documents(col1, docs)

    with col2:
        # Initialize chat history container
        chat_history_container = st.empty()

        # display chat
        display_chat_history(chat_history_container)

if __name__ == "__main__":
    main()