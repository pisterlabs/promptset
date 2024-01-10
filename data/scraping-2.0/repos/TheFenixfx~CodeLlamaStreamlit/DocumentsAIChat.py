from transformers import pipeline
import streamlit as st

from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


#Streamlit Initiation
st.set_page_config(
    page_title="Local Model querying with RAG",
    page_icon="",
    initial_sidebar_state="expanded",
    menu_items={"About": "Built bt JhonFx"},
)

if 'input_token_counter' not in st.session_state:
    st.session_state['input_token_counter'] = 0

if 'output_token_counter' not in st.session_state:
    st.session_state['output_token_counter'] = 0

# PATHS
repo_path = r"E:\Projects\deOlival2023\lastreviAGOST\de_olival_python\de_olival_python\home"
PATH = r"H:\oogabooga\text-generation-webui-1.6.1\models\codellama-13b-instruct.Q4_K_M\codellama-13b-instruct.Q4_K_M.gguf"

@st.cache_resource
def docloader():
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    len(documents)


    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents)
    len(texts)


    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity


    db = Chroma.from_documents(documents=texts, embedding=HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    ),persist_directory = r"E:\Projects\deOlival2023\\")


    retriever = db.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 6},
    )
    return retriever


def sidebar():
    """Configure the sidebar and user's preferences."""
    
    with st.sidebar.expander("🔧 SETTINGS", expanded=True):
        st.toggle('Cache Results', value=True, key="with_cache")
        st.toggle('Display Sources', value=True, key="with_sources")

    #st.sidebar.button('Clear Messages', type="primary", on_click=clear_chat_history) 
    st.sidebar.divider()
    with st.sidebar:
        "Documents and Tools"
        
def layout(llm,doc_db):
    """"Layout"""
    st.header("Chat with Documents using your own model")
    # System Messagec
    
    if "messages" not in st.session_state:    #session state dict is the way to navigate the state graphs that you are building
        print("function called")
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question"}
        ]

    # User input
    user_input = st.chat_input("Your question") # "Your Question" is only a placeholder and not actually a text input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
    
     # Generate response
    if st.session_state.messages[-1]["role"] != "assistant": # when the state is not assistant, because there is input, use the model
        try:
            generate_assistant_response(user_input,llm,doc_db)     
        except Exception as ex:
            st.error(str(ex))
    

def llmPrompInitialization():
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #system prompt
    system_prompt = """You are a helpful coder assistant, you will use the provided context to answer questions.
    Read the given code examples before answering questions and think step by step. If you can not answer a user question based on
    the provided context, inform the user. Do not use any other information for answer to the user"""

    instruction = """
    Context : {context}
    User: {question}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context","question"],
        template=header_template,
    )
    return QA_CHAIN_PROMPT


    
def generate_assistant_response(user_input, llm,doc_db):
    #Question about the documents
    docs = doc_db.get_relevant_documents(user_input)
    chain = load_qa_chain(llm, chain_type="stuff",prompt = llmPrompInitialization() )
    with st.chat_message("assistant"):
        with st.spinner("Working on request"):
            response = chain({"input_documents": docs, "question": user_input},return_only_outputs = True)
            message = {"role": "assistant", "content": response["output_text"]}
            st.write(response["output_text"]) #this is the actual text box on the browser
            st.session_state.messages.append(message["content"]) # after response we set the state again as it was to prevent infinite loop
            return response


@st.cache_resource # do not forget this decorator, streamlit is useful mostly because this to keep the model in memory
def llmini():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path=PATH, 
                    n_gpu_layers=43,
                    n_batch=512,
                    n_ctx=5000,
                    f16_kv=True,#thing in case
                    callback_manager=callback_manager,
                    verbose=True,
                    temperature=0.2)
    return llm


def main():
    """Set up user preferences, and layout"""
    llm = llmini()
    doc_db = docloader()
    #sidebar()
    layout(llm,doc_db)

if __name__ == "__main__":
    main()
