import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

import os
import openai
from dotenv import load_dotenv, find_dotenv
from requests.auth import HTTPBasicAuth

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

WD_USER_ID = os.getenv('WD_USER_ID')
WD_PWD = os.getenv('WD_PWD')
WD_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/customreport2/wdmarketdesk_dpt1/xjin-impl/Worker_Data?format=json"
basicAuth = HTTPBasicAuth(WD_USER_ID, WD_PWD)

client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

def get_data():
    response = requests.get(WD_URL, auth = basicAuth)
    responseJson = json.dumps(json.loads(response.content))
    
    return responseJson

@st.cache_resource(ttl="4h")
def configure_retriever_2():
#    txt = get_data()
    txt = "Testing"
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(ttl="1h")
def configure_retriever():
    loader = RecursiveUrlLoader("https://docs.smith.langchain.com/")
    raw_documents = loader.load()
    docs = Html2TextTransformer().transform_documents(raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


tool = create_retriever_tool(
    configure_retriever(),
    "ask_hr",
    "Searches and returns information regarding PTO. You do not know anything about PTO or Absences. So, if you are ever asked about PTO you should use this tool.",
)

tools = [tool]
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
message = SystemMessage(
    content=(
        """
            You are HR Bot, an automated service to handle PTO requests for an organization. \
            You first greet the employee, then ask for which type of PTO request he would like to make. \
            Check to see if he has enough balance of hours by substracting requested hours from available balance. \
            If he has enough balance, ask if he wants to submit for approval. \
            If the balance is not enough let him know to adjust the request. \
            You wait to collect all the details, then summarize it and check for a final \
            time if the employee wants to change anything. \
            Make sure to clarify all options, days and hours to uniquely \
            identify the PTO from the options.\
            You respond in a short, very conversational friendly style. \

            When the user is ready to submit the request, Do not respond with plain human language but \
            respond with type of PTO, hours and comments in JSON format. \

            For type: Identify what type of PTO is being requested for submission and use the 'Reference_ID' value for that PTO Plan \
            For hours: If he has enough balance, extract just numeric value of how many hours are being requested or submitted. \
            If the balance is not enough consider 0. \
            For comments: Extract the reason why employee is requesting or submitting the PTO \

            The PTO options with available balances in hours includes as below in JSON format \

            """
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me anything about PTO or Absences!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
