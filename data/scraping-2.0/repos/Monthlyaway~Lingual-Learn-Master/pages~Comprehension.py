import streamlit as st

from bigdl.llm.langchain.llms import TransformersLLM
from langchain import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain.schema import SystemMessage
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from pathlib import Path
# from langchain.callbacks.stdout import StdOutCallbackHandler

# from langchain.callbacks.base import BaseCallbackHandler

import re

st.set_page_config(
    page_title="Comprehension",
    page_icon="👋",
)


# Default model name
MODEL_NAME = ""


# Use cache to load model, no need to reload after web rerun
@st.cache_resource
def load_transformers_llm(model_name = MODEL_NAME):
    # Define the base folder path
    base_folder_path = "F:/Study/Code/llm-models"

    # Append MODEL_NAME to the folder path
    model_path = base_folder_path + "/" + model_name



    if (model_name == "lmsys-vicuna-7b-v1.5"):
        llm = TransformersLLM.from_model_id(
            model_id=model_path,
            model_kwargs={"temperature": 0.2, "trust_remote_code": True},
            
        )
    elif (model_name == "Llama-2-7b-chat-hf"):
        llm = TransformersLLM.from_model_id_low_bit(
            model_id=model_path,
            model_kwargs={"temperature": 0.2, "trust_remote_code": True},
            
        )

    return llm


st.title("Comprehension")

# User input model name and max new tokens, used to initialize llm
with st.sidebar:
    MODEL_NAME = st.selectbox(
        'Choose local model',
        ("lmsys-vicuna-7b-v1.5", "Llama-2-7b-chat-hf"),
        placeholder="Select...",
    )
    max_new_tokens = st.number_input("Set max new tokens")

    if MODEL_NAME == "":
        st.warning("Please select a model")
        st.stop()

    if not max_new_tokens:
        st.warning("Please input max new tokens")
        st.stop()


llm = load_transformers_llm(MODEL_NAME)
st.success("Model " + MODEL_NAME + " loaded, enjoy your journey")


tab1, tab2, tab3 = st.tabs(["Article 1", "Article 2", "Article 3"])

prompt_template = """ 
Analyze and break down the following paragraph. Provide a detailed breakdown, including the identification of key themes, concepts, and any notable examples or explanations within the paragraph. Answer in a markdown format.

{input}

Analyze:
"""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    llm_kwargs={"max_new_tokens":max_new_tokens}
    ,verbose=False
)

if "compre_tab1_msgs" not in st.session_state:
    st.session_state.compre_tab1_msgs = []

for message in st.session_state.compre_tab1_msgs:
    with tab1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if input := st.chat_input("What is up?"):
    st.session_state.compre_tab1_msgs.append({"role": "user", "content": input})
    with tab1:
        with st.chat_message("user"):
            st.markdown(input)
    
        with st.chat_message("assistant"):
            with st.spinner('Wait for it...'):
                response = llm_chain.predict(input = input)
            # Use regex to extract text after "Transformed Paragraph:"
            match = re.search(r'Analyze:(.*)', response, re.DOTALL)

            if match:
                transformed_text = match.group(1).strip()
            st.markdown(transformed_text)
        
    st.session_state.compre_tab1_msgs.append({"role": "assistant", "content": transformed_text})

with tab2:
    st.subheader("Will be supported in the future :point_up:")

with tab3:
    st.subheader("Will be supported in the future :point_up:")