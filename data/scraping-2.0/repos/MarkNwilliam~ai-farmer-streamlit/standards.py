from langchain.chat_models import ChatVertexAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Milvus
from langchain.document_loaders import WeatherDataLoader
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.zilliz import Zilliz
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.zilliz import Zilliz
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
#from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
from IPython.display import JSON
from google.cloud import aiplatform
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.globals import set_debug
from dotenv import load_dotenv

load_dotenv()

service_account_path = os.path.join(os.path.dirname(__file__), 'lablab-392213-7e18b3041d69.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

llm = ChatVertexAI()
set_debug(True)

"""
tru = Tru()
hugs = Huggingface()
openai = OpenAI()
tru.reset_database()

feedback_functions = [

Feedback(hugs.language_match).on_input_output()

]

f_relevance = Feedback(openai.relevance).on_input_output()

# Moderation metrics on output
f_hate = Feedback(openai.moderation_hate).on_output()
f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()
"""


def show():


    st.title("📝 Q&A Crop production")
    with st.spinner('Wait for it...'):

        # replace
        COLLECTION_NAME = os.getenv('COLLECTION_NAME')
        DIMENSION = int(os.getenv('DIMENSION'))
        ZILLIZ_CLOUD_URI = os.getenv('ZILLIZ_CLOUD_URI')
        ZILLIZ_CLOUD_USERNAME = os.getenv('ZILLIZ_CLOUD_USERNAME')
        ZILLIZ_CLOUD_PASSWORD = os.getenv('ZILLIZ_CLOUD_PASSWORD')
        ZILLIZ_CLOUD_API_KEY = os.getenv('ZILLIZ_CLOUD_API_KEY')
        PATH_TO_FILE = os.getenv('PATH_TO_FILE')
        connection_args = { 'uri': ZILLIZ_CLOUD_URI, 'token': ZILLIZ_CLOUD_API_KEY }
        path_to_file= "crop_production.pdf"

        loader = PyPDFLoader(path_to_file)

        docs = loader.load()

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=504, chunk_overlap=0)
        all_splits = text_splitter.split_documents(docs)



        embeddings = OpenAIEmbeddings()

        vector_store = Zilliz(embedding_function=embeddings,connection_args=connection_args,collection_name=COLLECTION_NAME,drop_old=False,
        ).from_documents(
            all_splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args=connection_args,
        )

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                         verbose=True)
        """
        tru_recorder = TruChain(qa,
                                app_id='Chain3_ChatApplication',
                                feedbacks=[

                                    Feedback(hugs.not_toxic).on_output(),
                                    Feedback(hugs.positive_sentiment).on_output(),
                                    Feedback(openai.relevance).on_input_output()
                                ]
                                )
       """
        st.success('Chat is ready')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How I can help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():



        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        """
        with tru_recorder as recording:
            result = qa({"query": prompt})
        """
        result = qa({"query": prompt})
        st.session_state.messages.append({"role": "assistant", "content": result["result"]})
        st.chat_message("assistant").write(result["result"])


