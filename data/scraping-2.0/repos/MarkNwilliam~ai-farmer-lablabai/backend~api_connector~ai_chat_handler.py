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
from trulens_eval import Feedback, Huggingface, Tru, TruChain, LiteLLM
from IPython.display import JSON
from google.cloud import aiplatform
import os

# Replace 'your-service-account-file.json' with the name of your JSON key file
service_account_path = os.path.join(os.path.dirname(__file__), '')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

os.environ["OPENWEATHERMAP_API_KEY"] = ""
os.environ["WOLFRAM_ALPHA_APPID"] = ""
os.environ["ALPHAVANTAGE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

tru = Tru()
hugs = Huggingface()
tru.reset_database()

feedback_functions = [

Feedback(hugs.language_match).on_input_output()

]

llm = ChatVertexAI()

tools = load_tools(["openweathermap-api", "wolfram-alpha"], llm)



# replace
COLLECTION_NAME = 'Aifarmer'
DIMENSION = 768
ZILLIZ_CLOUD_URI = "h"
ZILLIZ_CLOUD_USERNAME = ""
ZILLIZ_CLOUD_PASSWORD = ""
ZILLIZ_CLOUD_API_KEY = ""
connection_args = { 'uri': ZILLIZ_CLOUD_URI, 'token': ZILLIZ_CLOUD_API_KEY }
path_to_file= ""

loader = PyPDFLoader(path_to_file)

docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=504, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)



embeddings = OpenAIEmbeddings()

vector_store = Zilliz(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    drop_old=True,
    ).from_documents(
    all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args=connection_args,
    )





def process_ai_chat(user_input):
        """
        Process the user input using AI chat logic and return the AI's response.
        """
        query = "The AB shall be impartial to ?"
        docs = vector_store.similarity_search(query)
        print(docs)
        llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        retriever = vector_store.as_retriever()

        template = """You are a professional farmer and analyst who analysing the performance of the farming operations , when given a location find the weather and use it when your answering questions to help the farmer and always add emojis so they can clearly understand 
        {context}
        Question: {question}
        Helpful Answer:"""
        rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm2
            )

        print(rag_chain.invoke("Explain The AB shall be impartial to ?"))
        #weather = OpenWeatherMapAPIWrapper()


        #weather_data = weather.run("London,GB")
        #print(weather_data)
        # Prompt
        prompt = ChatPromptTemplate(
            messages=[
            SystemMessagePromptTemplate.from_template(
             "You are a professional farmer and analyst on farming get use the tools and knowldge you have to help the farmer "
             ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
            ]
            )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory )


        # Initialize LiteLLM-based feedback function collection class:
        litellm = LiteLLM(model_engine="chat-bison")

        # Define a relevance function using LiteLLM
        relevance = Feedback(litellm.relevance_with_cot_reasons).on_input_output()

        tru_recorder = TruChain(chain,
            app_id='Chain1_ChatApplication',
            feedbacks=[relevance])



        with tru_recorder as recording:
            response = chain({"question": user_input})



        print(tru.get_records_and_feedback(app_ids=[])[0] )




        ai_response = response["text"]
        return ai_response
