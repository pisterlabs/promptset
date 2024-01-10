# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from agents.SQLagent import build_sql_agent
from agents.csv_chat import build_csv_agent
from utils.utility import ExcelLoader
# app.py
from typing import List, Union, Optional
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
import os
import pandas as pd
from kafka import KafkaProducer

st.session_state.csv_file_paths = []

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""



def open_ai_key():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
@st.cache_data
def dbActive():
    os.environ['DB_ACTIVE'] = 'false'


def init_page() -> None:
    st.set_page_config(
    )
    st.sidebar.title("Options")
    icon, title = st.columns([3, 20])
    with icon:
        st.image('./img/image.png')
    with title:
        st.title('Finance Chatbot')
    st.session_state['db_active'] = False
def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI QA assistant. "
                    "When answering questions, use the context provided to you."
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    )
            )
        ]
        st.session_state.costs = []



def get_csv_file() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    import tempfile
    
    st.header("Upload Document or Connect to a Databse")
    
    uploaded_files = st.file_uploader(
        label="Here, upload your documents you want AskMAY to use to answer",
        type= ["csv", 'xlsx', 'pdf','docx'],
        accept_multiple_files= True
    )

    if uploaded_files:
        all_docs = []
        csv_paths = []
        all_files = []
        for file in uploaded_files:
            
            Loader = None
            if file.type == "text/plain":
                Loader = TextLoader
            elif file.type == "application/pdf":
                Loader = PyPDFLoader
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                Loader = Docx2txtLoader

            elif file.type == "text/csv":
                flp = './temp.csv'
                pd.read_csv(file).to_csv(flp, index=False)
                csv_paths.append(flp)

            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                loader = ExcelLoader(file)
                paths = loader.load()
                
                csv_paths.extend(paths)

            else:
                print(file.type)
                file.type
                raise ValueError('File type is not supported')

            if Loader:
                with tempfile.NamedTemporaryFile(delete=False) as tpfile:
                    tpfile.write(file.getvalue())
                    loader = Loader(tpfile.name)
                    docs = loader.load()
                    all_docs.extend(docs)

            #text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        if all_docs:
            documents = text_splitter.split_documents(all_docs)
            all_files.append(('docs', documents))
        if csv_paths:
            all_files.append(('csv', csv_paths))
        all_files = tuple(all_files)

        return all_files
    else:
        return None
    
def get_db_credentials(model_name, temperature, chain_mode='Database'):
    """
    creates a form for the user to input database login credentials
    """

    # Check if the form has already been submitted
    
    db_active = os.environ['DB_ACTIVE']
    if db_active == "true":
        print(db_active)

        return st.session_state['models']
        
    else:
        username = None
        host = None
        port = None
        db = None
        password = None
        import time
        pholder = st.empty()
        
        with pholder.form('Database_Login'):
            st.write("Enter Database Credentials ")
            username = st.text_input('Username').strip()
            password = st.text_input('Password', type='password',).strip()
            rdbs = st.selectbox('Select RDBS:',
                                ("Postgres",
                                'MS SQL Server/Azure SQL',
                                "MySQL",
                                "Oracle")
                            )
            port = st.number_input('Port')
            host = st.text_input('Hostname').strip()
            db = st.text_input('Database name').strip()

            submitted = st.form_submit_button('Submit')

        if submitted:
            with st.spinner("Logging into database..."):
                
                llm_chain, llm = init_agent(model_name=model_name,
                                    temperature=temperature,
                                    rdbs = rdbs,
                                    username=username,
                                    password=password,
                                    port=port,
                                    host=host,
                                    database=db,
                                    chain_mode = chain_mode)
            st.session_state['models'] = (llm_chain, llm)
            st.success("Login Success")
            os.environ['DB_ACTIVE'] = "true"
            db_active = os.environ['DB_ACTIVE']
            st.session_state['db_active'] = True
            time.sleep(2)
            pholder.empty()

            # If the form has already been submitted, return the stored models
        if db_active == "true":
            #return st.session_state['models']
            mds =  st.session_state['models']
            st.write("Reached")
            return mds
        else:
            st.stop()


def build_vector_store(
    docs: str, embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings]) \
        -> Optional[Qdrant]:
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).
    """
    
    if docs:
        with st.spinner("Loading FIle ..."):
            chroma = Chroma.from_documents(
             docs, embeddings
            )
    
        st.success("File Loaded Successfully!!")
    else:
        chroma = None
    return chroma


# Select model 

def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-0613",
                                   "gpt-3.5-turbo-16k-0613",
                                   "gpt-4",
                                   "text-davinci-003",
                                   ))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    chain_mode = st.sidebar.selectbox(
                        "What would you like to query?",
                        ("Documents", "CSV|Excel", 'Database')
    )
    #api_key  = st.sidebar.text_input('OPENAI API Key')
    
    return model_name, temperature, chain_mode,# api_key


def init_agent(model_name: str, temperature: float, **kwargs) -> Union[ChatOpenAI, LlamaCpp]:
    """
    Load LLM.
    """
    llm_agent = None  # Initialize llm_agent with a default value
    
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("text-dav"):
        llm =  OpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=f"./models/{model_name}.bin",
            input={"temperature": temperature,
                   "max_length": 2048,
                   "top_p": 1
                   },
            n_ctx=2048,
            callback_manager=callback_manager,
            verbose=False,  # True
        )
    chain_mode = kwargs['chain_mode']
    if chain_mode == 'Database':
        rdbs = kwargs['rdbs']
        username = kwargs['username']
        password = kwargs['password']
        host = kwargs['host']
        port = kwargs['port']
        database = kwargs['database']
        #print('----------------------------------------------------------------')
        #st.write(print(rdbs,username,password,host,port,database ))
        #print(rdbs,username,password,host,port,database )
        llm_agent = build_sql_agent(llm=llm, rdbs=rdbs, username=username, password=password,
                                    host=host, port=port, database=database)
    if chain_mode == 'CSV|Excel':
        file_paths = kwargs['csv']
        if file_paths is not None:
            with st.spinner("Loading CSV FIle ..."):
                llm_agent = build_csv_agent(llm, file_path=file_paths)
    
    return llm_agent, llm

def get_retrieval_chain(model_name: str, temperature: float, **kwargs) -> Union[ChatOpenAI, LlamaCpp]:
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("text-dav"):
        llm =  OpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=f"./models/{model_name}.bin",
            input={"temperature": temperature,
                   "max_length": 2048,
                   "top_p": 1
                   },
            n_ctx=2048,
            callback_manager=callback_manager,
            verbose=False,  # True
        )
    docsearch = kwargs['docsearch']
    retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever = docsearch.as_retriever(max_tokens_limit=4097)
            )
        
    return retrieval_chain, llm

def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings]:
    """
    Load embedding model.
    """
    if model_name.startswith("gpt-") or model_name.startswith("text-dav"):
        return OpenAIEmbeddings()
    elif model_name.startswith("llama-2-"):
        return LlamaCppEmbeddings(model_path=f"./models/{model_name}.bin")

def get_answer(llm_chain,llm, message) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    import langchain

    if isinstance(llm, (ChatOpenAI, OpenAI)):
        with get_openai_callback() as cb:
            try:
                if isinstance(llm_chain, RetrievalQAWithSourcesChain):
                    response = llm_chain(message)
                    answer =  str(response['answer'])# + "\n\nSOURCES: " + str(response['sources'])
                else:
                    answer = llm_chain.run(message)
            except langchain.schema.output_parser.OutputParserException as e:
                response = str(e)
                if not response.startswith("Could not parse tool input: "):
                    raise e
                answer = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return answer, cb.total_cost

def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content



def main() -> None:
    import openai
    init_page()
    dbActive()
    try:
        open_ai_key()
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        model_name, temperature, chain_mode = select_llm()
        embeddings = load_embeddings(model_name)
        files = get_csv_file()
        paths, texts, chroma = None, None, None

        if chain_mode == 'Database':
            llm_chain, llm = None, None
            try:
                print(os.environ['DB_ACTIVE'])
                if os.environ['DB_ACTIVE'] == "true":
                    llm_chain, llm = st.session_state['models']
                    
                else:
                    llm_chain, llm = get_db_credentials(model_name=model_name, temperature=temperature,
                                                    chain_mode=chain_mode)
            except KeyError:
                st.sidebar.warning('Provide a Database Log in Details')
                os.environ['DB_ACTIVE'] = "false"
                llm_chain, llm = get_db_credentials(model_name=model_name, temperature=temperature,
                                                    chain_mode=chain_mode)
                
                
                
            except Exception as e:
                err = str(e)
                st.error(err)
                

        elif files is not None:
            for fp in files:
                if fp[0] == 'csv':
                    paths = fp[1]
                elif fp[0] == 'docs':
                    texts = fp[1]
            if texts:
                import openai
                try:
                    chroma = build_vector_store(texts, embeddings)
                except openai.error.AuthenticationError:
                    st.echo('Invalid OPENAI API KEY')
            
            if chain_mode == "CSV|Excel":
                if paths is None:
                    st.sidebar.warning("Note: No CSV or Excel data uploaded. Provide atleast one data source")
                llm_chain, llm = init_agent(model_name, temperature, csv=paths, chain_mode=chain_mode)

            elif chain_mode == 'Documents':
                try:
                    assert chroma != None
                    llm_chain, llm = get_retrieval_chain(model_name, temperature, docsearch = chroma)
                except AssertionError as e:
                    st.sidebar.warning('Upload at least one document')
                    llm_chain, llm = None, None
                
            
        else:
            if chain_mode == "CSV|Excel":
                try: 
                    assert paths != None
                except AssertionError as e:
                    st.sidebar.warning("Note: No CSV data uploaded. Upload at least one csv or excel file")

            elif chain_mode == 'Documents':
                try:
                    assert chroma != None
                except AssertionError as e:
                    st.sidebar.warning('Upload at least one document or swith to data query')
                    
        

        init_messages()

        # Supervise user input
        st.header("Personal FinanceGPT")
        container = st.container()
        with container:
            
            user_input = st.chat_input("Input your question!")
            
        if user_input:
            try:
                assert type(llm_chain) != type(None)
                if chroma:
                    context = [c.page_content for c in chroma.similarity_search(
                        user_input, k=10)]
                    user_input_w_context = PromptTemplate(
                        template=PROMPT_TEMPLATE,
                        input_variables=["context", "question"]) \
                        .format(
                            context=context, question=user_input)
                    
                else:
                    user_input_w_context = user_input
                st.session_state.messages.append(
                    HumanMessage(content=user_input_w_context))
                
                
                with st.spinner("Assistant is typing ..."):
                    answer, cost = get_answer(llm_chain,llm, user_input)
                    st.write(answer)

                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)
            except AssertionError:
                st.warning('Please provide a context source')

        # Display chat history
        chat_history = []
        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                chat_history.append({'assistant' : message.content})
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                chat_history.append({'user': extract_userquesion_part_only(message.content)})
                with st.chat_message("user"):
                    st.markdown(extract_userquesion_part_only(message.content))

        # Create a Kafka producer instance with the provided configuration
        try:
            producer = KafkaProducer(bootstrap_servers='zkless-kafka-bootstrap:9092')
            
            # Define the topic name and key (modify as needed)
            topic_name = "tim-topic"
            key = "tim_key"

            # Print chat history and send to Kafka
            for entry in chat_history:
                for role, msg in entry.items():
                    print(f"{role.capitalize()}: {msg}")

            # Encode the message to bytes
                msg_encoded = msg.encode('utf-8')
    
            # Produce a message to the Kafka topic
                producer.send(topic_name, key=key, value=msg_encoded)

        # Ensure all messages are sent
            producer.flush()

        except Exception as e:
            print(f"Optional: Failed to send message to Kafka due to: {e}")

        costs = st.session_state.get("costs", [])
        st.sidebar.markdown("## Costs")
        st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            st.sidebar.markdown(f"- ${cost:.5f}")
    except openai.error.AuthenticationError as e:
        st.warning("Incorrect API key provided: You can find your API key at https://platform.openai.com/account/api-keys")
    except openai.error.RateLimitError:
        st.warning('OpenAI RateLimit: Your API Key has probably exceeded the maximum requests per min or per day')


# streamlit run app.py
if __name__ == "__main__":
    main()
