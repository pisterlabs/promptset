from typing import List
from ninja import Router
import urllib, requests
import os,sys
from dotenv import load_dotenv, find_dotenv
from llama_index import download_loader
from courtneyOracle.api.v1.utils.utils import *
from courtneyOracle.api.v1.utils.Notion.injest import *
from courtneyOracle.api.v1.utils.Notion.qa import *
from django.http import JsonResponse

#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
#from langchain import OpenAI
#from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
#from llama_index import LLMPredictor, ServiceContext
from llama_index import download_loader
#import json

#Find Get all the environment variables from .env files
load_dotenv(find_dotenv())

"""
Get current directory the unusual way
"""

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent) #append the parent directory from where the file is found.

router = Router()

@router.get("/youtube")
def transcribe_youtube(request,url:str):
    response = "Hi Baby Girl"
    #url = "https://www.youtube.com/watch?v=V264u5kFOTo"
    video_id = [url]
    print(YoutubeTranscriptReader(video_id).yt_link)
    current_Transcript_software = YoutubeTranscriptReader(video_id)

    # instantiate the class and call the member function:
    documents = current_Transcript_software.load_data(video_id)
    return documents

@router.get("/talktobooks")
def talk2books(request,name:str):
    response = "hi Baby Girl"
    response = ask_bot(input_index = 'index.json')
    print("\nBot says: \n\n" + response.response + "\n\n\n")
    return None

#TODO: Finish the Conversation with the Books Tomorrow. --> Marcus Aurelius

#Helper functions
class Convert2Json:
    def __init__(self,text):
        self.text = text

@router.get("/twitter")
def twitter(request,username:str):
    response = "hi Baby Girl"
    assert (isinstance(username,str)) #Collect stuffs to errors if you want. Further modification
    # All the keys here.
    twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    TwitterTweetReader = download_loader("TwitterTweetReader")
    loader = TwitterTweetReader(bearer_token=twitter_bearer_token)
    #example username: "davidasinclair"
    documents = loader.load_data(twitterhandles=[username])
    print (documents)
    processed_document = Convert2Json(text=documents)
    document_dict = {"text": str(processed_document.text).split("[Document(text=")[1]}
    #Pass the dictionary to the JsonResposnse
    return JsonResponse(document_dict)


# Add Option to get to Notion.so
@router.get("/notion")
def talk2Notion(request, question:str):
    response = "hi Baby Girl"

    # To be modified afterwards.
    try:
        injest_notion()
        print("successfully injested")
        response_dict = {"text": response, "success": True}
    except Exception as e:
        print(f"An error occurred: {e}")
        response_dict = {"text": f"An error occurred: {e}", "success": False}

    #Call the QA script here and convert it into
    #a dictionary.
    response_dict = {"text":qa(question)}
    # JSONResponse takes in a dictionary. We use {"text": output of api}
    return JsonResponse(content=response_dict)
