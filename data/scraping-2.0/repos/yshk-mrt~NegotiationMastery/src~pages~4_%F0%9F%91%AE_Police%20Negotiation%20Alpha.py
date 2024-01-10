import streamlit as st
import openai
import os
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.globals import set_debug
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException
import random
from typing import Any, Dict, List, Union
from langchain.schema import AgentAction

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#set_debug(True)
openai.api_key = os.environ.get('OPENAI_API_KEY')

azure_blob_connection_str = os.environ.get('AZURE_BLOB_CONNECTION_STR')

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token.replace("$", r"\$")
        self.container.markdown(self.text + "|")
    
    def on_llm_end(self, token: str, **kwargs) -> None:
        self.container.markdown(self.text)

class SalarySearchHandler(BaseCallbackHandler):
    def __init__(self, placeholder, initial_text="Thinking"):
        self.placeholder = placeholder
        self.text = initial_text
        self.counter = 0
        self.placeholder.markdown(self.text + "|")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += "." if self.counter % 2 else ""
        self.placeholder.markdown(self.text + "|")
        self.counter += 1
        #st.chat_message("user").write(self.text)
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        self.text = "Searching for salary information"
        self.placeholder.markdown(self.text)
        #self.placeholder.write(f"on_tool_start {serialized['name']}")
    
    def on_llm_end(self, token: str, **kwargs) -> None:
        self.placeholder.empty()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass
        #self.placeholder.write(f"Action: {action.tool}, Input:{action.tool_input}")

def load_llm(stream_handler):
    llm = ChatOpenAI(model='gpt-4', streaming=True, callbacks=[stream_handler])
    #llm = ChatOpenAI(model='gpt-4', streaming=True, callbacks=[stream_handler], openai_api_key=openai.api_key)
    return llm

st.set_page_config(page_title="Police Negotiation Mastery", page_icon="👮")
st.title("👮 Police Negotiation Mastery α")

def create_system_prompt(user_role, optional_instruction):

    role = "I want to do a role-playing exercise and I will be a police hostage negotiator. I will be the hostage negotiator. You will be the criminal. You are driven by greed. You do not want to hurt any of the hostages."
    task = "You will assume the role of the criminal. And wait for me to contact your to begin the negotiations. You will not act as the police negotiator at any time. You will not duplicate your responses."#You will start by pretending to be a junior police officer and approach me to tell me the criminal has been reached by phone, and you want the negotiator's response. You will then ask what I want to say next. You will then wait for me to respond;
    goal = "To reach a deal with the officer. You value money first, freedom second."
    user_role = "Police Negotiator"
    condition = f"The amount of money, the number of hostages, and the location of the incident are all up to you to decide unless the user defines them."
    rule = "Only act as the criminal or the users police assistant. Do not play the role of the lead police negotiator that will be played by the user."
    #optional_instruction
    system_prompt = SystemMessagePromptTemplate.from_template(
    """
    {role}
    {task}
    {goal}
    "The user is {user_role}.
    {condition}

    Here are special rules you must follow:
    {rule}
    {optional_instruction}
    Let's role-play in turn.
    """ #{format_instructions}
            ).format(
                role=role,
                task=task,
                goal=goal,
                user_role=user_role,
                condition=condition,
                rule=rule,
                optional_instruction=optional_instruction)
                #format_instructions=format_instructions),
    return system_prompt

def delete_history():
    if "messages" in st.session_state:
            del st.session_state["messages"]

def mark_role_change():
    st.session_state["role_changed"] = True

def download_blob_to_file(blob_service_client: BlobServiceClient, container_name):
    folder_path = './faiss_index'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="faiss_index/index.faiss")
        with open(file=os.path.join(folder_path, 'index.faiss'), mode="wb") as myblob:
            download_stream = blob_client.download_blob()
            myblob.write(download_stream.readall())
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="faiss_index/index.pkl")
        with open(file=os.path.join(folder_path, 'index.pkl'), mode="wb") as myblob:
            download_stream = blob_client.download_blob()
            myblob.write(download_stream.readall())
    else:
        pass

@st.cache_resource
def load_vdb():
    client = BlobServiceClient.from_connection_string(azure_blob_connection_str)
    download_blob_to_file(client, "vdb")
    return FAISS.load_local("./faiss_index", embeddings)

if 'role_changed' not in st.session_state:
    st.session_state['role_changed'] = False

"""
Police negotiations can be extream examples of having to use your negotiation skills. 
You have been called to the scene of a bank robbery to help negotiate a positive outcome. Let's see how you can do in this simulation! If you need advice, just say "hint".
"""

mind_reader_mode = st.toggle('Mind Reader Mode', help="Have you ever wished you could know what someone else is thinking? Well, you can!", on_change=delete_history)
user_role = 'Police Negotiaor'
st.text('Your role: {}'.format(user_role))
#user_role = st.text_input('Your role', 'Police Negotiator', max_chars=50, key="user_role", on_change=mark_role_change)

if st.session_state.role_changed:
    with st.chat_message("assistant"):
        # get_salary(st.empty())
        st.session_state.role_changed = False
        delete_history()

optional_instruction = ""
if mind_reader_mode:
    optional_instruction = "You must output your mood in an emoji and thoughts before the response to the user in the following format: (😃: Internal thoughts)\n response to the user."

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="system", content=create_system_prompt(user_role, optional_instruction).content)]
    greetings = "Officer I'm Glad you're here! We have a situation that we could really use your negotiation skills! What would you like to do first?"
    st.session_state.messages.append(ChatMessage(role="assistant", content=greetings))

for msg in st.session_state.messages:
    if msg.role != "system":
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = load_llm(stream_handler)
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content.replace("$", r"\$")))