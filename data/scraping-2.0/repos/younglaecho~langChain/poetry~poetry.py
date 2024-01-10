from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers

load_dotenv()

chat_model = ChatOpenAI()

# chat_model = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q3_K_L.bin",
#     model_type="llama"
# )

import streamlit as st

st.title('인공지능 시인')

content = st.text_input('시의 주제를 말씀해주세요', '코딩')

if st.button('시 작성 요청하기'):
    with st.spinner("시 작성 중"):
        result = chat_model.invoke(content + "에 대한 시를 써줄래?")
        st.write(result if type(result) == 'str' else result.content)
