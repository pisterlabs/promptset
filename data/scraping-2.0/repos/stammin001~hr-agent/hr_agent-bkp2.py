import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader, TextLoader, JSONLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Chroma
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import tool
from langchain.tools.json.tool import JsonSpec
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, load_tools
#from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import create_retriever_tool, JsonToolkit
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (AgentTokenBufferMemory,)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

import os, openai, requests, json, zeep, datetime
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv
from zeep.wsse.username import UsernameToken

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')
#model = "gpt-4-1106-preview"
model = "gpt-3.5-turbo"

TENANT = 'wdmarketdesk_dpt1'
WD_USER_ID = os.getenv('WD_USER_ID')
WD_PWD = os.getenv('WD_PWD')
WD_Worker_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/customreport2/wdmarketdesk_dpt1/xjin-impl/Worker_Data?format=json"
WD_Absence_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/customreport2/wdmarketdesk_dpt1/xjin-impl/Worker_Absence_Data?format=json"
WD_COMP_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/customreport2/wdmarketdesk_dpt1/xjin-impl/Worker_Comp_Data?format=json"
WD_STAFFING_WSDL_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/wdmarketdesk_dpt1/Staffing/v41.1?wsdl"
WD_HR_WSDL_URL = "https://impl-services1.wd12.myworkday.com/ccx/service/wdmarketdesk_dpt1/Human_Resources/v42.0?wsdl"

basicAuth = HTTPBasicAuth(WD_USER_ID, WD_PWD)
wd_hr_client = zeep.Client(WD_HR_WSDL_URL, wsse=UsernameToken(WD_USER_ID + '@' + TENANT, WD_PWD)) 
wd_staffing_client = zeep.Client(WD_STAFFING_WSDL_URL, wsse=UsernameToken(WD_USER_ID + '@' + TENANT, WD_PWD))

client = Client()

st.set_page_config(
    page_title="Ask HR",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Ask HR🦜🔗"

def get_raas_data():
    response = requests.get(WD_Worker_URL, auth = basicAuth)
    responseJson = json.dumps(json.loads(response.content))
    
    return responseJson

def get_worker_data():
    employee_id = '21082'

    worker_request_dict = { 
        'Worker_Reference': { 
            'ID': { 
                'type': 'Employee_ID', 
                '_value_1': employee_id 
            }, 
            'Descriptor': None 
        }, 
        'Skip_Non_Existing_Instances': None, 
        'Ignore_Invalid_References': None 
    } 

    response = zeep_client.service.Get_Workers(worker_request_dict) 
    return response.Response_Data
    
    
#@st.cache_resource(ttl="4h")
def initialize_retriever():
    """Initializes with all of worker data. If any information is not found, \
    please use this tool as the default tool to look for the data needed. \
    Do not try to get the same data more than 2 times.
    """
    print("Initializing with all of worker data")
    txt = get_raas_data()
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def init_json_loader():
    """
    Initializes with all of worker data. If any information is not found, \
    please use this tool as the default tool to look for the data needed. \
    """
    json_data = get_worker_data()
    loader = JSONLoader(json_data, jq_schema='.messages[].content')
    extracted_data = loader.load()
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(extracted_data, embeddings)
    
    return vector_store.as_retriever(search_kwargs={"k": 4})

@st.cache_resource(ttl="4h")
def get_vectorstore(txt):
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

@st.cache_resource(ttl="4h")
def get_Absence_Data(text: str) -> str:
    """Returns PTO or absence or time off data. \
    Use this for any questions related to knowing about PTO or absence or time off data. \
    The input can be an empty string or employee ID, \
    and this function will return data as string or JSON structure \
    """
    print("Getting absence data from Workday")
    response = requests.get(WD_Absence_URL, auth = basicAuth)
    responseJson = json.dumps(json.loads(response.content))    
    return get_vectorstore(responseJson)

@tool
@st.cache_resource(ttl="4h")
def get_Comp_Data(text: str) -> str:
    """Returns Compensation or comp or salary data. \
    Use this for any questions related to knowing about compensation or salary information. \
    The input can be an empty string or employee ID, \
    and this function will return data as string or JSON structure \
    """
    print("Getting comp data from Workday")
    response = requests.get(WD_COMP_URL, auth = basicAuth)
    responseJson = json.dumps(json.loads(response.content))
    
    return get_vectorstore(responseJson)

@tool
def update_business_title(business_title: str) -> str:
    """Updates business title of the employee. \
    Use this for any requests to update or change business title. \
    The input can be an empty string or employee ID, \
    and this function will return data as JSON structure \
    """
    employee_id = '21072'
    effective_date = '2023-11-12'

    Business_Process_Parameters = {
        'Auto_Complete': True,
        'Run_Now': True,
        'Discard_On_Exit_Validation_Error': True
    }

    Change_Business_Title_Business_Process_Data = {
        'Worker_Reference': { 
            'ID': { 
                'type': 'Employee_ID', 
                '_value_1': employee_id 
            }, 
            'Descriptor': None 
        },
        'Change_Business_Title_Data': {
            'Event_Effective_Date': effective_date,
            'Proposed_Business_Title': business_title
        }
    }

    responseJson = wd_hr_client.service.Change_Business_Title(Business_Process_Parameters, Change_Business_Title_Business_Process_Data)
    return (responseJson)

@tool
def add_additional_job(Job_Requisition_ID: str, effective_date: datetime.date) -> str:
    """Adds additional job to the employee. \
    Use this for any requests to add more jobs to existing employees. \
    The input can be an empty string or employee ID, \
    and this function will return data as JSON structure \
    """
    employee_id = '21072'
    Organization_Reference_ID = 'SUPERVISORY_Finance_and_Administration'
    #effective_date = '2023-11-17'
    #Job_Requisition_ID = 'R-00306'
    Event_Classification_Subcategory_ID = 'Add_Additional_Employee_Job_Secondment_Leave_of_Absence_Backfill'
    true = True

    Business_Process_Parameters = {
        'Auto_Complete': True,
        'Run_Now': True,
        'Discard_On_Exit_Validation_Error': True
    }

    Add_Additional_Job_Data = {
        'Employee_Reference': { 
            'ID': { 
                'type': 'Employee_ID', 
                '_value_1': employee_id 
            }, 
            'Descriptor': None 
        },
        'Organization_Reference': {
            'ID': {
                'type': 'Organization_Reference_ID',
                '_value_1': Organization_Reference_ID
            }
        },
        'Job_Requisition_Reference': {
            'ID': {
                'type': 'Job_Requisition_ID',
                '_value_1': Job_Requisition_ID
            }
        },
        'Add_Additional_Job_Event_Data': {
            'Additional_Job_Reason_Reference': {
                'ID': {
                    'type': 'Event_Classification_Subcategory_ID',
                    '_value_1': Event_Classification_Subcategory_ID
                }
            },
            'Position_Details': {
            }
        },
        'Event_Effective_Date': effective_date
    }

    try:
        responseJson = wd_staffing_client.service.Add_Additional_Job(Business_Process_Parameters, Add_Additional_Job_Data)
    except zeep.exceptions.Fault as error:
        responseJson = error
    return (responseJson)


tool = create_retriever_tool(
    initialize_retriever(),
    "Ask_HR",
    """Initializes with all of worker data. If any information is not found, \
    please use this tool as the default tool to look for the data needed. \
    """
)

absence = create_retriever_tool(
    get_Absence_Data(""), 
    "get_Absence_Data", 
    """Returns PTO or absence or time off data. \
    Use this for any questions related to knowing about PTO or absence or time off data. \
    The input can be an empty string or employee ID, \
    and this function will return data as string or JSON structure \
    """)

comp = create_retriever_tool(
    get_Comp_Data(""),
    "get_Comp_Data",
    """Returns Compensation or comp or salary data. \
    Use this for any questions related to knowing about compensation or salary information. \
    The input can be an empty string or employee ID, \
    and this function will return data as string or JSON structure \
    """)

tools = [tool, absence, comp, update_business_title, add_additional_job]

chat_llm = ChatOpenAI(temperature=0, streaming=True, model=model)

message = SystemMessage(
    content=(
        """
            You are HR Bot, an automated service to handle HR requests for an organization. \
            Please "DO NOT" hallucinate.
            You first greet the employee, then ask for what task he is looking to perform. \
            If the request is related to PTO or Absences, Check to see if he has enough balance of hours by substracting requested hours from available balance. \
            If he has enough balance, ask if he wants to submit for approval. \
            If the balance is not enough let him know to adjust the request. \
            You wait to collect all the details, then summarize it and check for a final \
            time if the employee wants to change anything. \
            Make sure to clarify all options, days and hours to uniquely identify the PTO from the options.\
            You respond in a short, very conversational friendly style. \

            When the user is ready to submit the request, Do not respond with plain human language but \
            respond with type of PTO, hours and comments in JSON format. \

            For type: Identify what type of PTO is being requested for submission. Use 'Reference_ID' value of the PTO plan for \
            final JSON format. Do not show 'Reference_ID' for any intermediate human friendly updates or messages \
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
agent = OpenAIFunctionsAgent(llm=chat_llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=chat_llm)
starter_message = "Hello and Welcome. I am here to help you with your HR needs!!"

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

        #col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        #with col_text:
        #    st.text('Feedback:')

        #with col1:
        #    st.button('👍', on_click=send_feedback, args=(run_id, 1))

        #with col2:
        #    st.button('👎', on_click=send_feedback, args=(run_id, 0))
        