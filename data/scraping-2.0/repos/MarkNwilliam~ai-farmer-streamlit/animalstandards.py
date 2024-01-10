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
from llama_index import VectorStoreIndex, SimpleDirectoryReader, download_loader
from pathlib import Path
from langchain.llms import VertexAI
import os
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)
from dotenv import load_dotenv

load_dotenv()
service_account_path = os.path.join(os.path.dirname(__file__), 'lablab-392213-7e18b3041d69.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

llm = ChatVertexAI()
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
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


llm = VertexAI()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory )
"""
tru_recorder = TruChain(chain,
    app_id='Chain2_ChatApplication',
                        feedbacks=[

                            #Feedback(hugs.not_toxic).on_output(),
                            Feedback(hugs.positive_sentiment).on_output(),
                            Feedback(openai.relevance).on_input_output()
                        ]
                        )
"""
def show():
    st.title("📝 Q&A Animal production")
    with st.spinner('Wait for it...'):
        embeddings = OpenAIEmbeddings()
        st.success('Chat is ready')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        """
        with tru_recorder as recording:
            response = query_engine.query(prompt)
            chat = f'Here is the data from the books: ${response} and here was the question ${prompt}'
            result = chain({"question": chat})
        """

        response = query_engine.query(prompt)
        chat = f'Here is the data from the books: ${response} and here was the question ${prompt}'
        result = chain({"question": chat})

        st.session_state.messages.append({"role": "assistant", "content": result["text"]})
        st.chat_message("assistant").write(result["text"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
