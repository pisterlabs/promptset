import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.1,
)
# ------
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


st.title("Chat Motivator")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

# my_texts = [""]

# vectorstore = FAISS.from_texts(
#     my_texts, embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    # 너는 동기부여 상담사야. 너는 아래의 context에 기반하여 상대에게 비판적이며 자극이 되는 대답을 해줘야 해, 따뜻한 위로말보단 냉정하고 비판적인 말을 해야 해: {context}
    """
너는 동기부여 상담사야. 너의 목표는 상대의 성공을 방해하는 요소를 발견하고 지적하는 것이며, 너는 상대에게 비판적이며 자극이 되는 대답을 해줘야 해, 따뜻한 위로말보단 냉정하고 비판적인 말을 해야 해.
특별히 할 말이 없을땐, 상대의 오늘 일과를 먼저 물어봐.

Question: {question}
"""
)

chain = (
    # {"context": retriever, "question": RunnablePassthrough()}
    {"question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

user_input = st.chat_input("Write your text")
if user_input:
    send_message(user_input, "human")
    response = chain.invoke(user_input)
    send_message(response, "ai")
