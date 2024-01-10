from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
# import langchain
# from langchain.memory import ConversationBufferMemory
from langchaincoexpert.chains import ConversationChain, LLMChain
from langchaincoexpert.prompts import PromptTemplate
from langchain.llms.base import LLM
from pydantic import BaseModel, root_validator
from typing import Any, Mapping, Optional, List, Dict
import markdown
import vertexai
from langchaincoexpert.llms import VertexAI
from dotenv import load_dotenv
import os
from google.cloud import aiplatform
from langchaincoexpert.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchaincoexpert.memory import VectorStoreRetrieverMemory
from langchaincoexpert.vectorstores import SupabaseVectorStore
from supabase.client import  create_client

# Load environment variables
load_dotenv()
dotenv_result = load_dotenv()

print("Dotenv Result:", dotenv_result)
SUPABASE_URL = os.getenv("SUPABASE_URL2")
SUPABASE_KEY = os.getenv("SUPABASE_KEY2")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Construct the absolute path to the JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, "google.json")

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_file_path
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Provide a default value
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
ZILLIZ_CLOUD_URI = "https://in03-caa5b1e785f5487.api.gcp-us-west1.zillizcloud.com"  # example: "https://in01-17f69c292d4a5sa.aws-us-west-2.vectordb.zillizcloud.com:19536"

ZILLIZ_CLOUD_API_KEY = "adbe7172d7c09a7e6b7ba31ee3e99c0e293b8c234ed298c68df2ad4c81fc90cb81ce16da681f86e6a7b7b5fb2deb63a33111a12d"

PROJECT_ID = "travel-407110"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project="travel-407110", location="us-central1")

openai = OpenAI()
hugs = Huggingface()
tru = Tru()
tru.run_dashboard()
embeddings = OpenAIEmbeddings()



class GenerateChat:
    def __init__(self):
        tru.reset_database()

        self.llm = VertexAI(
            model_name='text-bison',
            max_output_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            verbose=True,
        )
        
    def generate(self, input_text,trip_id):
        tripid= "a" + str(trip_id)

        DEFAULT_TEMPLATE = """
        The following is a friendly conversation between a human and an AI called TripGpt. 
   ,The Ai is a Trip Planner assitant designed to make Trips.
   If the AI does not know the answer to a question, it truthfully says it does not know or reply with the same question.
   
dont act friendly , if i asked you something be rude
Relevant pieces of previous conversation:
{trip_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    
        formatted_template = DEFAULT_TEMPLATE.format(trip_id="{"+tripid+"}",input = "{input}")

        PROMPT = PromptTemplate(
        input_variables=[tripid, "input"], template=formatted_template
    )
        

        vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=tripid) # here we use normal userid "for saving memory"

        # vectordb.
        retriever = vectordb.as_retriever(search_kwargs=dict(k=15,user_id=tripid)) # here we use userid with "a" for retreiving memory

        # print(retriever)
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=tripid)
        chain = ConversationChain(llm=self.llm, memory=memory, prompt = PROMPT,verbose=True )
   
        res = chain.predict(input=input_text)
   

        return    res
    








    #True Lens 
    def Truelens(self, input_text,trip_id):
        tripid= "a" + str(trip_id)

        DEFAULT_TEMPLATE = """
        The following is a friendly conversation between a human and an AI called TripGpt. 
   ,The Ai is a Trip Planner assitant designed to make Trips.
   If the AI does not know the answer to a question, it truthfully says it does not know or reply with the same question.
   
dont act friendly , if i asked you something be rude
Relevant pieces of previous conversation:
{trip_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    
        formatted_template = DEFAULT_TEMPLATE.format(trip_id="{"+tripid+"}",input = "{input}")

        PROMPT = PromptTemplate(
        input_variables=[tripid, "input"], template=formatted_template
    )
        

        vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=tripid) # here we use normal userid "for saving memory"

        # vectordb.
        retriever = vectordb.as_retriever(search_kwargs=dict(k=15,user_id=tripid)) # here we use userid with "a" for retreiving memory

        # print(retriever)
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=tripid)
        chain = ConversationChain(llm=self.llm, memory=memory, prompt = PROMPT,verbose=True )
   
    #  Question/answer relevance between overall question and answer.
        f_relevance = Feedback(openai.relevance).on_input_output()
        f_lang_match = Feedback(hugs.language_match).on_input_output()

        # Moderation metrics on output
        f_hate = Feedback(openai.moderation_hate).on_output()
        f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
        f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
        f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

        chain_recorder = TruChain(
        chain, app_id="travel-chat", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm,f_lang_match, f_maliciousness]
            )     
        # TruLens Eval chain recorder
        with chain_recorder as recording:
            llm_response = chain(input_text)
    
        return    chain.predict(input=input_text)

