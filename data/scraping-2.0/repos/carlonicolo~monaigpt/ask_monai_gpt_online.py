"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
import os
import re
from langchain.llms import OpenAI
import api_key as key

os.environ['OPENAI_API_KEY'] = key.OPENAI_API_KEY

def get_bot():
    # Create the embeddings
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory='./monai_gpt_db', embedding_function=embeddings)

    #Prediction part
    bot_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                        vectordb, return_source_documents=True)
    
    return bot_qa

def extract_code_text(answer):
    # Extract code block from message
    code_block_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    code_blocks = code_block_pattern.findall(answer)

    # Remove code blocks from the message
    message_clean = code_block_pattern.sub('', answer)

    return message_clean, code_blocks

def get_text():
    st.header("How can I assist you")
    input_text = st.text_input("", "Give me a definition of MONAI?", key="input")
    return input_text



# From here down is all the StreamLit UI.
st.set_page_config(page_title="MONAI GPT", page_icon=":robot:")
st.header("MONAI :robot_face: GPT - Demo")
st.image('utils/banner.png')
st.markdown('For specific information about the MONAI documentation, you can visite the website [here](https://docs.monai.io/en/stable/api.html).')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "generated_code" not in st.session_state:
    st.session_state["generated_code"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = get_text()

if user_input:
    bot_qa = get_bot()
    result = bot_qa({"question": user_input, "chat_history": ""})

    message_clean, code_blocks = extract_code_text(result['answer'])
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(message_clean)
    st.session_state.generated_code.append(code_blocks)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        # Display the extracted code block(s) with formatting
        message(st.session_state["generated"][i], key=str(i))
        for code_block in st.session_state["generated_code"][i]:
            st.code(code_block.strip())