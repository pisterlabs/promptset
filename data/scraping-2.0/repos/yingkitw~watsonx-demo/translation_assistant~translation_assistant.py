import os
from dotenv import load_dotenv
import streamlit as st

from langchain.chains import LLMChain
from langchain import PromptTemplate

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams

load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

creds = Credentials(api_key,api_endpoint)

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=200,
    min_new_tokens=1,
    stream=False,
    temperature=0.7,
    top_k=50,
    top_p=1
).dict()

with st.sidebar:
    st.title("Translation Assistant")

st.title("Translation Assistant")

text_input = st.text_area('Enter text')
    
# Create a selectbox for the user to select the target language
target_language = st.selectbox('Select language', [ 'English', 'Spanish', 'French', 'German','Chinese','Korean','Japanese','Hindi'])

# Create a button that the user can click to initiate the translation process
translate_button = st.button('Translate')

# Create a placeholder where the translated text will be displayed
translated_text = st.empty()

# Handle the translation process when the user clicks the translate button
if translate_button:
    translated_text.text('Translating...')
    llm = LangChainInterface(model="bigscience/mt0-xxl",credentials=creds, params=params)
    prompt = PromptTemplate(template=f"Translate '{text_input}' to {target_language}",
                            input_variables=[])
    chain = LLMChain(llm=llm,prompt=prompt)
    response_text = chain.predict()
    translated_text.text(response_text)
