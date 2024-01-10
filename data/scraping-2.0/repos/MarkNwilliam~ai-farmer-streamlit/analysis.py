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
from dotenv import load_dotenv
load_dotenv()
service_account_path = os.path.join(os.path.dirname(__file__), 'lablab-392213-7e18b3041d69.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

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



llm = ChatVertexAI()


tools = load_tools(["openweathermap-api", "wolfram-alpha"], llm)


load_dotenv()  # Load variables from .env

# Access variables
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
DIMENSION = int(os.getenv('DIMENSION'))
ZILLIZ_CLOUD_URI = os.getenv('ZILLIZ_CLOUD_URI')
ZILLIZ_CLOUD_USERNAME = os.getenv('ZILLIZ_CLOUD_USERNAME')
ZILLIZ_CLOUD_PASSWORD = os.getenv('ZILLIZ_CLOUD_PASSWORD')
ZILLIZ_CLOUD_API_KEY = os.getenv('ZILLIZ_CLOUD_API_KEY')
connection_args = { 'uri': ZILLIZ_CLOUD_URI, 'token': ZILLIZ_CLOUD_API_KEY }
path_to_file= "crop_production.pdf"

loader = PyPDFLoader(path_to_file)

docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=504, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)



embeddings = OpenAIEmbeddings()

""""
vector_store = Zilliz(embedding_function=embeddings,connection_args=connection_args,collection_name=COLLECTION_NAME,drop_old=True,
).from_documents(
    all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args=connection_args,
)

query = "The AB shall be impartial to ?"
docs = vector_store.similarity_search(query)

print(docs)
"""

llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#retriever = vector_store.as_retriever()

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt = PromptTemplate.from_template(template)
"""
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm2
)
"""

#print(rag_chain.invoke("Explain The AB shall be impartial to ?"))
#weather = OpenWeatherMapAPIWrapper()


#weather_data = weather.run("London,GB")
#print(weather_data)
# Prompt
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
    app_id='Chain1_ChatApplication',
                        feedbacks=[

                            #Feedback(hugs.not_toxic).on_output(),
                            Feedback(hugs.positive_sentiment).on_output(),
                            Feedback(openai.relevance).on_input_output()
                        ]
                        )
"""

print("Welcome to the ChatBot Console App!")
print("Enter 'exit' to quit the app.")
#1EYGP229IBI5SCJI

#4AYU3P-LWG9VGPY89

""""
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    with tru_recorder as recording:
        response = chain({"question": user_input})



    print("AI:", response["text"])

    #print(tru.get_records_and_feedback(app_ids=[])[0] )

"""

def show():
    # Streamlit frontend
    st.title("Contextual Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Record with TruLens
            """
            with tru_recorder as recording:
                full_response = chain.run(prompt)
            """
            full_response = chain.run(prompt)
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})



