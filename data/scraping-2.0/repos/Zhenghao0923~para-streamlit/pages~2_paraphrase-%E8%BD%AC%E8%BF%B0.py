import streamlit as st
import openai
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import time

API_KEY=os.environ['OPENAI_API_KEY']
prompt_key=os.environ['prompt_key']
# if"openai_mode" not in st.session_state:
#     st.session_state["openai_model"] ="gpt-3.5-turbo"

# if "message" not in st.session_state:
#     st.session_state.message=[]
    
# #display chat messages from history on app return
# for message in st.session_state.message:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

st.title("转述-Paraphrase")


# prompt=st.chat_input("请提供需要转写的段落")
# if prompt:
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.message.append({"role":"user","content":prompt})
    
#     with st.chat_message("assistant"):
#         message_placeholder=st.empty()
#         full_response=""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role":m["role"],"content":m["content"]}
#                 for m in st.session_state.message
#             ],
#             stream=True,
#         ):
#             full_response+=response.choice[0].delta.get("content","")
#             message_placeholder.markdown(full_response+"")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role":"assistant","content":full_response})

llm=OpenAI(openai_api_key=API_KEY, temperature=1)

para_prompt_template=PromptTemplate(
    template=prompt_key+"{texts}",
    input_variables=['texts']
)
user_input=st.text_input("提供需要复述的段落")
para_chain=LLMChain(
    llm=llm,
    prompt=para_prompt_template,
    verbose=True
)
if st.button("生成转写") and user_input:
    
    with st.status("上传段落……"):
        st.write("理解文意……")
        time.sleep(2)
        st.write("重构段落……")
        time.sleep(1)
        output=para_chain.run(texts=user_input)
    col1,col2=st.columns(2)
    with col1:
        st.header("新版本")
        st.write(output)
    with col2:
        st.header("旧版本")
        st.write(user_input)
