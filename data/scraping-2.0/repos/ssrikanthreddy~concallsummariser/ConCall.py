import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


class HumanMessage:
    def __init__(self, content):
        self.content = content

    def to_dict(self):
        return {"type": "human", "content": self.content}

    @classmethod
    def from_dict(cls, message_dict):
        return cls(content=message_dict["content"])

def main():
    # Create two tabs
    tabs = ["Home", "PDF Summerizer", "PDF Chat"]
    selected_tab = st.sidebar.radio("Select a tab", tabs, key="tabs")

    if selected_tab == "Home":
        home()
    elif selected_tab == "PDF Summerizer":
        summary()
    elif selected_tab == "PDF Chat":
        chat()

# Sidebar contents
with st.sidebar:
    st.title('Welcome to ConCall Summeriser!')
    add_vertical_space(5)

load_dotenv()

def home():
    if "VectorStore" not in st.session_state:
        st.session_state.VectorStore = None

    st.header("Upload a PDF to chat with")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                st.session_state.VectorStore = pickle.load(f)
        else:
            with st.spinner(text="Vectorizing PDF..."):
                embeddings = OllamaEmbeddings(model="mistral")
                st.session_state.VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(st.session_state.VectorStore, f)

def summary():
    st.header("PDF Summerizer")
    
    if "summary" not in st.session_state:
        st.session_state.summary = False
     # Accept user questions/query
    query = "Summerise the following conference call in detail, use a table in markdown for each question and answer. Then also create a bulleted point summary of the whole transcript"
    if query and st.session_state.VectorStore:
        docs = st.session_state.VectorStore.similarity_search(query=query, k=3)

        llm = Ollama(
            model="mistral",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
  
        response = chain.run(input_documents=docs, question=query)
         

        # Store the chat results in session_state
        if "ChatResults" not in st.session_state:
            st.session_state.ChatResults = []
        
        st.session_state.ChatResults.append(response)

       
        for result in st.session_state.ChatResults:
            st.write(result)

        if st.button("Download Summary as Markdown"):
            download_text = "\n\n".join(result for result in st.session_state.ChatResults)
            st.download_button(
                label="Download",
                data=download_text,
                file_name="summary.md",
                mime="text/markdown"
            )

def chat():
    if "memory" not in st.session_state:
        st.session_state.memory = None
    bruh = st.session_state.get('messages', [])
    chat = Ollama(
            model="mistral",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
    chat, 
    st.session_state.VectorStore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)
  
       

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Chat with PDF 🤖")
   
    user_input = st.text_input("Your message: ", key="user_input")

    # handle user input
    if user_input:
        bruh = st.session_state.get('messages', [])
        docs = st.session_state.VectorStore.similarity_search(query=user_input, k=3)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        with st.spinner("Thinking..."):
            response = chain({"question": user_input})
            
        st.session_state.messages.append(AIMessage(content=response['answer']))
    
    
    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()
