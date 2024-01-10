import os
from dotenv import load_dotenv
import streamlit as st

from langchain.llms.openai import OpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

st.title('LangChain Text Summariser with Watsonx')

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

load_dotenv()
api_key = os.getenv("API_KEY", None)
project_id = os.getenv("PROJECT_ID", None)

creds = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "apikey" : api_key
}

params = {
    GenParams.DECODING_METHOD:"sample",
    GenParams.MAX_NEW_TOKENS:100,
    GenParams.MIN_NEW_TOKENS:1,
    GenParams.TEMPERATURE:0.5,
    GenParams.TOP_K:50,
    GenParams.TOP_P:1
}

space_id    = None
verify      = False

source_text = st.text_area("Source Text",height=200)

if st.button("Summarize"):
    if not source_text.strip():
        st.write(f"Please complete the missing fields")
    else:
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(source_text)

        docs = [Document(page_content=t) for t in texts[:3]]
        model = Model("google/flan-ul2",creds, params, project_id)
        llm = WatsonxLLM(model)
        chain = load_summarize_chain(llm,chain_type="map_reduce")
        summary = chain.run(docs)
        st.write(summary)