import streamlit as st
st.title('시를 짓는 인공지능')

#from langchain.chat_models import ChatOpenAI
# Chat_model = ChatOpenAI()

from langchain.llms import CTransformers

llm = CTransformers(
    model ='llama-2-7b-chat.ggmlv3.q2_K.bin',
    model_type = 'llama'
)

# text_input() 샤용해보기
content = st.text_input('시의 주제를 제시해 주세요')

if st.button('시 작성 요청하기'):
    with st.spinner('시를 작성 중입니다.'):
        result = llm.predict( 'write o poem about' + content + ':')
        st.write(result)