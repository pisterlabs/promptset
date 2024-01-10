import requests
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import ast


def answer_question(docs, user_question, useAzure):
    # add Azure gpt-4 api support in the future
    if useAzure:
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        os.environ["OPENAI_API_BASE"] = "https://user1-create-gpt.openai.azure.com/"
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")
        chat = AzureChatOpenAI(deployment_name="gpt-4",
                               openai_api_version="2023-03-15-preview")
        # chat = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_version="2023-03-15-preview")
    else:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        chat = ChatOpenAI(temperature=0.1)

    messages = [SystemMessage(
        content="You are a helpful bot for doctor. If needed, you can use the following document to help you answer the question in traditional Chinese.")]

    # Add the relevant documents to the chat context
    for i, doc in enumerate(docs):
        messages.append(SystemMessage(content=f"Document {i+1}: {doc}"))

    messages.append(HumanMessage(content=user_question))

    response = chat(messages).content
    reference = ""
    # messages.append(AIMessage(content=response))
    if docs:  # if there are any documents
        for i, doc in enumerate(docs):
            reference += f"### 文件 {i+1}\n"
            reference += doc.page_content  # Replace 'page_content' with the appropriate key for the document content
            reference += "\n\n"
    return response, reference
    


def classify_drug(user_question, useAzure):
    try:
        if useAzure:
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
            os.environ["OPENAI_API_BASE"] = "https://user1-create-gpt.openai.azure.com/"
            os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")
            chat = AzureChatOpenAI(
                deployment_name="gpt-4", openai_api_version="2023-03-15-preview")
            # chat = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_version="2023-03-15-preview")
        else:
            os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
            chat = ChatOpenAI(temperature=0.1)

        messages = [SystemMessage(
            content="You are a helpful bot for doctor. Please extract the medications of the texts and output in the format of ['drug1', 'drug2', 'drug3']. No other explanation is needed. If there is no medication, output:[]")]

        messages.append(HumanMessage(content=user_question))

        response = chat(messages).content
        # messages.append(AIMessage(content=response))

        return response

    except Exception as e:
        st.error(f"Error answering question: {e}")
        return "I'm sorry, I couldn't process your question. Please try again."


def search_documents(drugs):
    load_dotenv()
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY_2')
    PINECONE_API_ENV = 'asia-southeast1-gcp-free'
    INDEX_NAME = "nhi-v2"
    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

    # Load Pinecone index
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings)

    docs = []
    drugs = ast.literal_eval(drugs)
    if len(drugs) > 0:
        for drug in drugs:
            drug_docs = docsearch.similarity_search(
                drug, include_metadata=True)
            if len(drugs) > 4:
                docs.extend(drug_docs[0])
            else:
                docs.extend(drug_docs[0:2])
    return docs


def display_chat_history(chat_history_container):
    with chat_history_container:
        chat_history_html = "<div style='height: 500px; overflow-y: scroll;'>"
        st.header("Chat History")
        for msg in st.session_state.chat_history:
            chat_history_html += (
                f"<div style='text-align: right; background-color: #d0f0c0; border-radius: 5px; margin: 10px 10px 10px 50%; padding: 5px;'>{msg['user']}</div>"
            )
            chat_history_html += (
                f"<div style='text-align: left; background-color: #d3d3d3; border-radius: 5px; margin: 10px 50% 10px 10px; padding: 5px;'>{msg['response']}</div>"
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


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    
    # Load the .env file located in the project directory
    load_dotenv()
    useAzure = True

    drugs = classify_drug(user_input, useAzure=useAzure)
    docs = search_documents(drugs)

    response, reference = answer_question(docs, user_input, useAzure=useAzure)
    st.session_state['reference'] = reference
    st.session_state.generated.append(response)

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


def main():
    st.session_state.setdefault('past', [])
    st.session_state.setdefault('generated', [])
    if 'reference' not in st.session_state:
        st.session_state['reference'] = ''

    st.title("健保規定詢問")

    chat_tab, ref_tab= st.tabs(["Chat", "Reference"])

    with chat_tab:
        chat_placeholder = st.empty()
        with chat_placeholder.container():    
            for i in range(len(st.session_state['generated'])):                
                message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
                message(st.session_state['generated'][i], key=f"{i}")

        with st.container():
            st.text_input("User Input:", on_change=on_input_change, key="user_input")
            st.button("Clear message", on_click=on_btn_click)

    with ref_tab:
        st.markdown(st.session_state['reference'])

if __name__ == "__main__":
    main()