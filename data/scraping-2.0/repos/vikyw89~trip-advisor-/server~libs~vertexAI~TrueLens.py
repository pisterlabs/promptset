from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
# import langchain
# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from pydantic import BaseModel, root_validator
from typing import Any, Mapping, Optional, List, Dict
import markdown
import vertexai
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from google.cloud import aiplatform
from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import  create_client
from google.cloud import texttospeech
# from google.cloud import speech_v1p1beta1
from google.cloud import speech
# from google.cloud import speech
from libs.vertexAI.AudioConversion import mp3_to_wav_bytes
import traceback

# Load environment variables
load_dotenv()
dotenv_result = load_dotenv()

# client = texttospeech.TextToSpeechClient()

print("Dotenv Result:", dotenv_result)
SUPABASE_URL = os.getenv("SUPABASE_URL2")
SUPABASE_KEY = os.getenv("SUPABASE_KEY2")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Construct the absolute path to the JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, "google-credentials.json")

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_file_path
# cred = service_account.Credentials.from_service_account_file("google-credentials.json")
# print(cred)
client = speech.SpeechClient()

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
        self.memory = ConversationBufferMemory()

        
#     def generate(self, input_text,trip_id):
#         tripid= "a" + str(trip_id)

#         DEFAULT_TEMPLATE = """
#         The following is a friendly conversation between a human and an Translator AI . 
#    The AI translates any message sent to him to the desired language.
   
# Relevant pieces of previous conversation:
# {trip_id}
# (You do not need to use these pieces of information if not relevant)

# Current conversation:
# Human: {input}
# AI:"""
    
#         formatted_template = DEFAULT_TEMPLATE.format(trip_id="{"+tripid+"}",input = "{input}")

#         PROMPT = PromptTemplate(
#         input_variables=[tripid, "input"], template=formatted_template
#     )
        

#         vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=tripid) # here we use normal userid "for saving memory"

#         # vectordb.
#         retriever = vectordb.as_retriever(search_kwargs=dict(k=15,user_id=tripid)) # here we use userid with "a" for retreiving memory

#         # print(retriever)
#         memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=tripid)
#         chain = ConversationChain(llm=self.llm, memory=memory, prompt = PROMPT,verbose=True )
   
#         res = chain.predict(input=input_text)
     
#         return    res
    

    # def Translate(self,input , lan):
    #     """Synthesizes speech from the input string of text."""


    #     input_text = texttospeech.SynthesisInput(text=input)

    # # Note: the voice can also be specified by name.
    #     voice = texttospeech.VoiceSelectionParams(
    #         language_code="en-US",
    #         name="en-US-Studio-O",
    #     )

    #     audio_config = texttospeech.AudioConfig(
    #         audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    #         speaking_rate=1
    #     )

    #     response = client.synthesize_speech(
    #         request={"input": input_text, "voice": voice, "audio_config": audio_config}
    #     )

    #     # The response's audio_content is binary.
    #     with open("output.mp3", "wb") as out:
    #         out.write(response.audio_content)
    #         print('Audio content written to file "output.mp3"')
    def SpeechToText(self, audioz):
            # audio_content = mp3_to_linear16(audio)
            # print(audio_content)
            # audio
                        # s = get_mp3_sampling_rate(au)
                    
            # The content of the audio file to transcribe
        audio_content = audioz

            # transcribe speech
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
        encoding= speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code="en-US",
                model="default",
                audio_channel_count=1,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
            )

            # Detects speech in the audio file
        operation = client.long_running_recognize(config=config, audio=audio)
        print(operation)
        print("Waiting for operation to complete...")
        response = operation.result(timeout=90)
        print(response.results)

        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
        return(response.results)

    #True Lens 
#     def Truelens(self, input_text,trip_id):
#         tripid= "a" + str(trip_id)

#         DEFAULT_TEMPLATE = """
#         The following is a friendly conversation between a human and an AI called TripGpt. 
#    ,The Ai is a Trip Planner assitant designed to make Trips.
#    If the AI does not know the answer to a question, it truthfully says it does not know or reply with the same question.
   
# dont act friendly , if i asked you something be rude
# Relevant pieces of previous conversation:
# {trip_id}
# (You do not need to use these pieces of information if not relevant)

# Current conversation:
# Human: {input}
# AI:"""
    
#         formatted_template = DEFAULT_TEMPLATE.format(trip_id="{"+tripid+"}",input = "{input}")

#         PROMPT = PromptTemplate(
#         input_variables=[tripid, "input"], template=formatted_template
#     )
        

#         vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=tripid) # here we use normal userid "for saving memory"

#         # vectordb.
#         retriever = vectordb.as_retriever(search_kwargs=dict(k=15,user_id=tripid)) # here we use userid with "a" for retreiving memory

#         # print(retriever)
#         memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=tripid)
#         chain = ConversationChain(llm=self.llm, memory=memory, prompt = PROMPT,verbose=True )
   
#     #  Question/answer relevance between overall question and answer.
#         f_relevance = Feedback(openai.relevance).on_input_output()
#         f_lang_match = Feedback(hugs.language_match).on_input_output()

#         # Moderation metrics on output
#         f_hate = Feedback(openai.moderation_hate).on_output()
#         f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
#         f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
#         f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

#         chain_recorder = TruChain(
#         chain, app_id="travel-chat", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm,f_lang_match, f_maliciousness]
#             )       
#         with chain_recorder as recording:
#             llm_response = chain(input_text)

#         return   chain.predict(input=input_text)



    def generate(self, input_text):

        chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)
        # Question/answer relevance between overall question and answer.
        f_relevance = Feedback(openai.relevance).on_input_output()
        f_lang_match = Feedback(hugs.language_match).on_input_output()

        # Moderation metrics on output
        f_hate = Feedback(openai.moderation_hate).on_output()
        f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
        f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
        f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()
# TruLens Eval chain recorder
        chain_recorder = TruChain(
        chain, app_id="travel-chatz", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm,f_lang_match, f_maliciousness]
            )       
        with chain_recorder as recording:
            llm_response = chain(input_text)

        return       chain.predict(input=input_text)