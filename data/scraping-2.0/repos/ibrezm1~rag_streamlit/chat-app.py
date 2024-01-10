import vertexai
import os
from langchain.llms import VertexAI

PROJECT_ID = 'zeta-yen-319702'
REGION = 'us-central1'
BUCKET = 'gs://zeta-yen-319702-temp/'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './svc-gcp-key.py'

vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET
)

# https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#:~:text=PaLM%202%20for%20Text%20(%20text%2Dunicorn%20),with%20complex%20natural%20language%20tasks.


llm = VertexAI(
    #model_name="text-bison@001",
    model_name="text-unicorn",
    max_output_tokens=256,
    temperature=0.8,
    top_p=0.8,
    top_k=5,
    verbose=False,
)

from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage,AIMessage

system = "You are a helpful assistant who translate English to French"
human = "Translate this sentence from English to French. I love programming."
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chat = ChatVertexAI(model_name="codechat-bison", max_output_tokens=1000, temperature=0.5)

question = ''
retriever = ''
chat_history = ''
qa_prompt = ''

crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt}) 
result = crc({'question': question, 'chat_history': chat_history})


import streamlit as st




with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()

    #client = llm(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    
    #response = llm(prompt=prompt)

    vmessages = [] 
    vmessages.append(SystemMessage(content = "bot" ))
    for msg in st.session_state.messages[1:]:
        if (msg["role"] == "assistant"):
            vmessages.append(AIMessage(content = msg["content"] ))
        else:
            vmessages.append(HumanMessage(content = msg["content"]))

    print(vmessages)
    response = chat(
            vmessages
    )

    #msg = response.choices[0].message.content
    msg = response.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)