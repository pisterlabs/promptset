import os
from transformers import pipeline

from langchain.llms import HuggingFaceHub
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, 
                               AIMessagePromptTemplate, ChatPromptTemplate)
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import wolfram_alpha
from langchain.agents import load_tools, initialize_agent

import streamlit as st
from utils import build_llm, compute_cost, completion_smart_goal, memory_to_pandas

# TODO: Reorganize and structure code
# TODO: Add text input area in the side for system prompt testing
# TODO: Include chat history in the model input
# TODO: Modularize LLM model selection
# TODO: Separate completion and beta chat goal setting tabs

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

smart_gen = memory_df['AI_profiles'][5]

# --------- Local Utils --------- #
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def chat_smart_goal(smart_profile: dict,
                    system_prompt: str,
                    report: str,
                    human_input: str,
                    chat_history: str):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        smart_profile: dict
        provider: str
    return:
        smart_goal: str
    '''
    #name = smart_profile['name']
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']

    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_template = HumanMessagePromptTemplate.from_template("{input}")
    ai_template = AIMessagePromptTemplate.from_template("{response}")

    # create the list of messages
    chat_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template,
        ai_template
    ])
    
    # build chat llm
    chatgpt = build_llm(provider='ChatGPT4', 
                        max_tokens=max_tokens, temperature=temperature)
    
    # Build chain
    conversation_chain = ConversationChain(llm=chatgpt, prompt=chat_prompt,
                                           memory=ConversationBufferMemory())
    
    # Create a dictionary to hold the chat context
    chat_context = {'report': report, 'input': human_input, 'response': "Hi! I'm your wellness coach. I'm here to help you set a SMART goal. Shall we define the first dimension?"}
    
    # Add chat history to the context
    chat_context['chat_history'] = chat_history
    
    # Run chat LLM
    llm_output = conversation_chain.run(chat_context)
    
    # cost of report
    llm_cost = compute_cost(len(llm_output.split()), 'gpt-4')
    
    return llm_output, llm_cost

# Function for generating LLM response
def generate_response(prompt_input, message_history):

    llm = build_llm(150, 1, 'openai')

    #for dict_message in st.session_state.messages:
    #    string_dialogue = "You are a helpful assistant."
    #    if dict_message["role"] == "user":
    #        string_dialogue += "User: " + dict_message["content"] + "\n\n"
    #    else:
    #        string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    #prompt = f"{string_dialogue} {prompt_input} Assistant: "
    
    # Create the complete prompt with conversation history
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])
    prompt = f"{chat_history}\n\nUser: {prompt_input}\nAssistant: "
    
    return llm(prompt)
# -------------------------------- #

st.set_page_config(page_title="🤗💬 SmartBot Test")

with st.sidebar:
    st.title('🤗💬 SmartBot')
    st.subheader('Powered by 🤗 Language Models')
    st.text_area("Enter your system prompt here", height=400)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
            
# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User-provided prompt
# We isntantiate a new prompt with each chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, st.session_state.messages) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
