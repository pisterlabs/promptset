import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers


# chat_model = ChatOpenAI()
llm= CTransformers(
    model="llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama"
)
st.title('AI Poet')

content = st.text_input('Please provide topic of poem.')
if st.button('Request my poem'):
    with st.spinner('Poet is writting your poem'):
        result = llm.predict("Write poem about "+ content)
        st.write(result)



