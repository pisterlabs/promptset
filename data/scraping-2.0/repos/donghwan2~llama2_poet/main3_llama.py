import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers

llm = CTransformers(
    model = "llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type = "llama"
)

st.title('AI Poet')

content = st.text_input('What is topic?')

if st.button('Request a poem'):
    with st.spinner('Writing...'):
        result = llm.predict("write a poem about" + content + ": ")
        st.write(result)







