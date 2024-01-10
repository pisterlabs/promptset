from typing import List, Union
from langchain.vectorstores.chroma import Chroma
from langchain.callbacks import get_openai_callback
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.schema import Memory as StreamlitChatMessageHistory
from langchain.llms import CTransformers
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from time import sleep
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile
import warnings
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
import os
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import (PyPDFLoader, Docx2txtLoader, CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
APP_NAME = "ValonyLabsz"
MODEL = "gpt-3.5-turbo"
PAGE_ICON = ":rocket:"
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)
av_us = '/https://raw.githubusercontent.com/achilela/main/Ataliba' 
av_ass = '/https://raw.githubusercontent.com/achilela/main/Robot'
st.title(":rocket: Agent Lirio :rocket:")
st.markdown("I am your Technical Assistant ready to do all of the leg work on your documents, emails, procedures, etc.\
    I am capable to extract relevant info and domain knowledge!")




@st.cache_resource(ttl="1h")
def init_page() -> None:
    st.sidebar.title("Options")
def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            ]
        st.session_state.costs = []
user_query = st.chat_input(placeholder="Ask me Anything!")
def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    model_name = st.sidebar.radio("Choose LLM:", ("gpt-3.5-turbo-0613", "gpt-4", "llama-2"), key="llm_choice")
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name, streaming=True
)
    elif model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return CTransformers(model="/home/ataliba/LLM_Workshop/Experimental_Lama_QA_Retrieval/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_1.bin",
                                model_type="llama",
                                max_new_tokens=512,
                                temperature=temperature)
        

#openai_api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = os.getenv("OPENAI_API_KEY")

#openai_api_key = os.environ[OPENAI_API_KEY]
#openai_api_key = "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"


def configure_qa_chain(uploaded_files):
    docs = []
    if uploaded_files:
      if "processed_data" not in st.session_state:
          documents = []
      for file in uploaded_files:
           temp_filepath = os.path.join(os.getcwd(), file.name) # os.path.join(temp_dir.name, file.name)
           with open(temp_filepath, "wb") as f:
             f.write(file.getvalue())
           if temp_filepath.endswith((".pdf", ".docx", ".txt")):  #if temp_filepath.lower() == (".pdf", ".docx", ".txt"):
              loader = UnstructuredFileLoader(temp_filepath)
              loaded_documents = loader.load() #loader = PyPDFLoader(temp_filepath)
              docs.extend(loaded_documents) #loader.load_and_split())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# storing embeddings in the vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    persist_directory = "/home/ataliba/LLM_Workshop/Experimental_Lama_QA_Retrieval/db/"
    
    
    
   
    memory = ConversationBufferMemory(
    memory_key="chat_history", output_key='answer', return_messages=False)    
    
   
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")
    def on_retriever_start(self, query: str):  #def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")
    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)
uploaded_files = st.sidebar.file_uploader(
    label="Upload your files", accept_multiple_files=True,type=None
)
if not uploaded_files:
    st.info("Please upload your documents to continue.")
    st.stop()
retriever = configure_qa_chain(uploaded_files)
memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
llm = select_llm() # model_name="gpt-3.5-turbo"
qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory) #retriever=retriever, memory=memory)#, verbose=False
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Please let me know how can I be of a help today?"}]
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message(msg["role"]): #,avatar=av_us):
            st.markdown(msg["content"])
    else:
        with st.chat_message(msg["role"]): #,avatar=av_ass):
            st.markdown(msg["content"])
            
if user_query: #
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        message_placeholder =  st.empty()
        full_response = ""
        cb = PrintRetrievalHandler(st.container())
        response = qa_chain.run(user_query, callbacks=[cb])
        resp = response.split(" ")
        for r in resp:
             full_response = full_response + r + " "
             message_placeholder.markdown(full_response + "▌")
             sleep(0.1)
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
