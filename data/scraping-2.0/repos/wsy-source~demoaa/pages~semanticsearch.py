from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import SemanticField, SemanticConfiguration,SemanticSearch,SemanticPrioritizedFields
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import SemanticField, SemanticConfiguration,SemanticSearch,SemanticPrioritizedFields
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.messages import HumanMessage
# Import required libraries  
import os  

from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (

    VectorizedQuery,
)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import time

from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        time.sleep(0.020)
        self.container.chat_message("assistant").write(self.text,unsafe_allow_html=True)


model="gpt-4-32k"
embeddings_model="text-embedding-ada-002"

AZURE_SEARCH_ENDPOINT="https://gptdocument.search.windows.net"
ADMIN_KEY="W7XVOLsbEZgcvWqLD0MxpdVbAfWf1ZIZG7HYdIVSgPAzSeDTMtqR"
INDEX_NAME="gptdocument"

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, model="text-embedding-ada-002"):
    client = AzureOpenAI(
        api_key = "3397748fcdcb4a5fbeb6c2eb5a6a284f",  
        api_version = "2023-05-15",
        azure_endpoint = "https://sean-aoai-gpt4.openai.azure.com/"
    )
    return client.embeddings.create(input = [text], model=model).data[0].embedding

col1, col2 = st.columns([0.8,0.2])

with st.sidebar:
    count = st.number_input(label="Document Count",value=10)
    


question = col1.text_input(label="input you question")
result = col2.button("search")
con = st.container()
data= con.empty()
if result:
    prompt = """
    User Requirements: {user_requirements}
    Document: {document}
    """
    llm=AzureChatOpenAI(
        openai_api_type="azure",azure_endpoint="https://sean-aoai-gpt4.openai.azure.com/",
        api_key="3397748fcdcb4a5fbeb6c2eb5a6a284f",api_version="2023-05-15",
        model=model,callbacks=[StreamHandler(data)],streaming=True
    )
    credential=AzureKeyCredential(ADMIN_KEY)
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, INDEX_NAME, credential=credential)
    vector_query = VectorizedQuery(vector=generate_embeddings(question), k_nearest_neighbors=count, fields="contentVector")
    
    content= search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["content","page"]    )
    template= PromptTemplate.from_template(template=prompt)
    document = list(content)
    chain = LLMChain(prompt=template,llm=llm,verbose=True)
    chain.run(document=document,user_requirements=question)



    



