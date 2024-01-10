import streamlit as st
from time import  sleep
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# Avatar emojis
av_us = "🐵" 
av_ass = '🚽'

@st.cache_resource
def get_llm():
    # Model initialization
    checkpoint = "./model/"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                        device_map='auto',
                                                        torch_dtype=torch.float32)
    # LangChain plumbing code
    llm = HuggingFacePipeline.from_model_id(model_id=checkpoint,
                                            task = 'text2text-generation',
                                            model_kwargs={"temperature":0.7,"min_length":30, "max_length":350, "repetition_penalty": 5.0})
    
    return llm
    

template = """{text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chat = LLMChain(prompt=prompt, llm=get_llm())


st.title("LLM Nonsense ChatBot 🤡")
st.subheader("Ask me anything and I'll produce some utter garbage answer in mostly perfect English")

# Show sidebar with model selection
st.sidebar.title("What is this?")
# Write a description
st.sidebar.markdown("This is a proof of concept for a small LLM ([Large Language Model](https://en.wikipedia.org/wiki/Large_language_model)) deployed on the [Streamlit Community Cloud](https://streamlit.io/cloud). The application is using a small HuggingFace LLM model, [LaMini-Flan-T5-77M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-77M/tree/main), and does work as a chatbot, but whatever questions the user asks will be answered with mostly nonsense.")
st.sidebar.subheader("Why does it matter?")
st.sidebar.markdown("The model code is hosted entirely on Streamlit Community Cloud without making any calls to external servers. No API key needed, no external dependencies.")
st.sidebar.subheader("Why is it useless?")
st.sidebar.markdown("The model is very small (~300MB) to be able to run on Streamlit Community Cloud, but the results cannot be compared to large LLMs like ChatGPT4 (hundreds of GBs of data), etc. It also serves as an example that whatever LLM you're using, never blindly trust the output.")

# Set a default model
if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = "MBZUAI/LaMini-Flan-T5-248M"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])

# Accept user input
if myprompt := st.chat_input("Go ahead and ask me whatever...!"):
    # Display user message in chat message container
    with st.chat_message("user", avatar=av_us):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=av_ass):
        message_placeholder = st.empty()
        full_response = ""
        res  =  chat.run(myprompt)
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"

st.markdown("GitHub repo: [https://github.com/tjaensch/llm_nonsense_chatbot](https://github.com/tjaensch/llm_nonsense_chatbot)")