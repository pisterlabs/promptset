import streamlit as st
from streamlit_chat import message

from langchain.agents.agent_toolkits import GmailToolkit
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import ZeroShotAgent,AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.prompts import MessagesPlaceholder

import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse, parse_qs

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import os
os.environ['OPENAI_API_KEY'] = st.secrets["openapi_key"]

zapier_client_id = st.secrets["zapier_client_id"]
zapier_redirect_uri = st.secrets["zapier_redirect_uri"]
zapier_client_secret = st.secrets["zapier_client_secret"]
azure_token = st.secrets["azure_token"]

def init():
        
    st.set_page_config(
        page_title='Smart Mailbox Assistant',
        page_icon='🤖'
    )

def on_btn_click():
    del st.session_state.messages[:]
    
def positive_feedback():
    a = 1
    st.session_state.messages_alr_screen = True
    st.session_state.feedback_type = "Positive"
    
def negative_feedback():
    a = 1
    st.session_state.messages_alr_screen = True
    st.session_state.feedback_type = "Negative"
    
def main():
    init()
    
    st.header('Smart Mailbox Assistant 🤖')
    
    if "run_main" not in st.session_state:
        st.session_state.run_main = True
        
    if "access_token" not in st.session_state:
        st.session_state.access_token = ""
        
    if "agent_not_created" not in st.session_state:
        st.session_state.agent_not_created = True
        
    if "agent_chain" not in st.session_state:
        st.session_state.agent_chain = ""
        
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You're a helpful Mailbox Assistant")]    
        
    if "messages_alr_screen" not in st.session_state:
        st.session_state.messages_alr_screen = False

    if 'feedback_type' not in st.session_state:
        st.session_state.feedback_type = None

    if st.session_state.run_main:
        auth_url = f"https://nla.zapier.com/oauth/authorize/?response_type=code&client_id={zapier_client_id}&redirect_uri={zapier_redirect_uri}&scope=nla%3Aexposed_actions%3Aexecute"
        st.markdown(f"[Please click here to authenticate]({auth_url})")

        url = st.text_input('Once you have authenticated, please enter the landing page URL here:', key='url_input')

        if url:

            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            code = query_params.get('code', [''])[0]

            token_url = "https://nla.zapier.com/oauth/token/"
            data = {
                "code": code,
                "grant_type": "authorization_code",
                "client_id": zapier_client_id,
                "client_secret": zapier_client_secret,
                "redirect_uri": zapier_redirect_uri
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(token_url, data=data, headers=headers)
            st.session_state.access_token = response.json().get("access_token")
            refresh_token = response.json().get("refresh_token")
            if st.session_state.access_token != None:
                st.write('Authenticated ✅')
                        
            st.session_state.run_main = False
        
            if st.session_state.agent_not_created:
                
                try:
                    zapier = ZapierNLAWrapper(zapier_nla_oauth_access_token=st.session_state.access_token)
                    zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    
    
                    zapier_description = """|
                    A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "what did Dustin send me last time". 
                    Fetch the identified emails, open each one, and read its contents. 
                    This includes the body, subject line, sender information, and any attachments. Please decode and store any attachments found in the emails for later reference.
                    After reading the emails, extract key information, then present it in a clear, user-friendly format, as if you were responding to a second person: so instead of saying "he sent me an email..." you should say "he sent you an email...".
                    Each tool will have params associated with it that are specified as a list. 
                    You MUST take into account the params when creating the instruction. For example, 
                    if the params are [\'Search_String\'], your instruction should be something like "Find the latest emails from Dustin". 
                    Do not make up params, they will be explicitly specified in the tool description. 
                    If you do not have enough information to fill in the params, just say "not enough information provided in the instruction, missing <param>". 
                    If you get a none or null response, STOP EXECUTION, do not try to use another tool! This tool is specifically used for: Gmail: Find Email, and has params: [\'Search_String\']'
                    """
    
                    zapier_base_prompt = """
                    A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, 
                    for example "get the latest email from my bank" or "what did Dustin send me last time". 
                    Fetch the identified emails, open each one, and read its contents. 
                    This includes the body, subject line, sender information, and any attachments. Please decode and store any attachments found in the emails for later reference.
                    After reading the emails, extract key information, then present it in a clear, user-friendly format, as if you were responding to a second person: so instead of saying "he sent me an email..." you should say "he sent you an email...".
                    Each tool will have params associated with it that are specified as a list. 
                    You MUST take into account the params when creating the instruction. For example, 
                    if the params are [\'Search_String\'], your instruction should be something like "Find the latest emails from Dustin". 
                    Do not make up params, they will be explicitly specified in the tool description. 
                    If you do not have enough information to fill in the params, just say "not enough information provided in the instruction, missing <param>". 
                    If you get a none or null response, STOP EXECUTION, do not try to another tool!
                    This tool specifically used for: {zapier_description}, and has params: {params}'
                    """
    
                    zapier_toolkit.tools[1].description = zapier_description
                    zapier_toolkit.tools[1].base_prompt = zapier_base_prompt
    
                    suffix="""
                    Here's the past conversation:
                    {chat_history}
                    Question: {input}
                    Thought: {agent_scratchpad}
                    """
    
                    prompt = ZeroShotAgent.create_prompt(
                        [zapier_toolkit.get_tools()[1]],
                        prefix='Answer the following questions as best you can. You have access to the following tools:',
                        suffix=suffix,
                        input_variables=["input", "chat_history", "agent_scratchpad"],
                    )
                    memory = ConversationBufferMemory(memory_key="chat_history")
    
                    llm_chain = LLMChain(llm=OpenAI(temperature=0),  prompt=prompt)
    
                    agent_2 = ZeroShotAgent(llm_chain=llm_chain, tools=[zapier_toolkit.get_tools()[1]])
                    st.session_state.agent_chain = AgentExecutor.from_agent_and_tools(agent=agent_2, tools=[zapier_toolkit.get_tools()[1]], memory=memory)
    
                    llm_agent_instruc = """Answer the following questions as best you can. You have access to the following tools:\n\nGmail: Find Email: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "get the latest email about Jamaica". Fetch the identified emails, open each one, and read its contents. This includes the body, subject line, sender information, and any attachments. Please decode and store any attachments found in the emails for later reference.After reading the emails, extract key information, then present it in a clear, user-friendly format as if you were responding to a second person: so instead of saying "he sent me an email..." you should say "he sent you an email...". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are [\'Search_String\'], your instruction should be something like \'find the latest email from my bank\'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to use another tool! This tool is specifically used for: Gmail: Find Email, and has params: [\'Search_String\']\'\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Gmail: Find Email]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\n\nHere\'s the past conversation:\n{chat_history}\nQuestion: {input}\nThought: {agent_scratchpad}\n"""
    
                    st.session_state.agent_chain.agent.llm_chain.prompt.template = llm_agent_instruc
                    st.write('Agent created')
                    
                    st.session_state.agent_not_created = False
                except:
                    st.write("Please enter a valid URL. Make sure there's a code value in it.")
            
    with st.sidebar:
        user_input = st.text_input('What would you like to know?', key='user_input')   

    if st.session_state.messages:

        if st.session_state.messages_alr_screen == False:
            
            try:
                if user_input:
                    st.session_state.messages.append(HumanMessage(content=user_input))
                    with st.spinner('Thinking...'):
                        response = st.session_state.agent_chain(user_input)
                    st.session_state.messages.append(AIMessage(content=response['output']))
    
                messages = st.session_state.get('messages', [])
                for i, msg in enumerate(messages[1:]):
                    if i % 2 == 0:
                        message(msg.content, is_user=True, key=str(i) + '_user')
                    else:
                        message(msg.content, is_user=False, key=str(i) + '_ai')  
            except:
                st.write("Please make sure to first authenticate.")

            if len(st.session_state.messages) >= 3:
                col1, col2 = st.columns([1, 15])
                with col1:
                    thumbs_up = st.button("👍", key='loop_up', on_click=positive_feedback)
                with col2:
                    thumbs_down = st.button("👎", key='loop_down', on_click=negative_feedback)

        else:

            messages = st.session_state.get('messages', [])
            for i, msg in enumerate(messages[1:]):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + '_user')
                else:
                    message(msg.content, is_user=False, key=str(i) + '_ai')  

            if len(st.session_state.messages) >= 3:

                if st.session_state.feedback_type is None:

                    col1, col2 = st.columns([1, 15])
                    with col1:
                        thumbs_up = st.button("👍", key='loop_up_first', on_click=positive_feedback)
                    with col2:
                        thumbs_down = st.button("👎", key='loop_down_first', on_click=negative_feedback)
                        
                else:
                    
                    col1, col2 = st.columns([1, 15])
                    with col1:
                        thumbs_up = st.button("👍", key='loop_up_second', on_click=positive_feedback)
                    with col2:
                        thumbs_down = st.button("👎", key='loop_down_second', on_click=negative_feedback)
                    text_feedback = st.text_input("[Optional] Provide additional feedback")
                    if text_feedback:
                        messages_as_string = "<br/><br/>".join([("User: " if idx % 2 == 0 else "AI: ") + i.content for idx, i in enumerate(messages[1:])])
                        all_feedback = "<br/><br/><br/>".join([messages_as_string, text_feedback])
                        url = "https://dev.azure.com/kmeleon/Gen%20AI%20Onboarding%20V1/_apis/wit/workitems/$Feature?api-version=6.0"
                        headers = {
                            'Content-Type': 'application/json-patch+json',
                        }
                        data = [
                            {
                                "op": "add",
                                "path": "/fields/System.Title",
                                "value": "Chatbot Feedback"
                            },
                            {
                                "op": "add",
                                "path": "/fields/System.Description",
                                "value": all_feedback
                            },
                            {
                                "op": "add",
                                "path": "/fields/System.Tags",
                                "value": st.session_state.feedback_type
                            }
                        ]
                        response = requests.post(url, headers=headers, json=data, auth=HTTPBasicAuth('', azure_token))
                        if response.status_code == 200:
                            st.success('Thank you for your feedback!')
                            st.session_state['feedback_type'] = None 
                        else:
                            st.error('Failed to submit feedback.')
                            st.session_state['feedback_type'] = None 
                        st.session_state.messages_alr_screen = False
                    

        col1, col2, col3 = st.columns(3)

        with col2:
            if len(st.session_state.messages) >= 3: 
                st.button("Clear conversation history", on_click=on_btn_click)

    else:
        st.write('Chat deleted successfully!')
        st.session_state.messages = [SystemMessage(content="You're a helpful Mailbox Assistant")] 
        

if __name__ == '__main__':
    main()
